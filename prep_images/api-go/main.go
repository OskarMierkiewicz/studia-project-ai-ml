package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type Config struct {
	S3Endpoint  string
	S3AccessKey string
	S3SecretKey string
	S3UseSSL    bool

	PredBucket  string
	KeyTemplate string
}

func mustConfig() Config {
	endpoint := getenv("MINIO_ENDPOINT", "minio.mlops-infra.svc.cluster.local:9000")
	access := getenv("AWS_ACCESS_KEY_ID", "minio")
	secret := getenv("AWS_SECRET_ACCESS_KEY", "minio12345")

	useSSL := false
	if strings.ToLower(os.Getenv("MINIO_USE_SSL")) == "true" {
		useSSL = true
	}

	bucket := getenv("PREDICTIONS_BUCKET", "predictions")
	keyTpl := getenv("PREDICTIONS_KEY_TEMPLATE", "{ticker}/H{h}/predictions_test.csv")

	return Config{
		S3Endpoint:  endpoint,
		S3AccessKey: access,
		S3SecretKey: secret,
		S3UseSSL:    useSSL,
		PredBucket:  bucket,
		KeyTemplate: keyTpl,
	}
}

func newMinio(cfg Config) (*minio.Client, error) {
	return minio.New(cfg.S3Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.S3AccessKey, cfg.S3SecretKey, ""),
		Secure: cfg.S3UseSSL,
	})
}

func keyFromTemplate(tpl, ticker string, h int) string {
	out := strings.ReplaceAll(tpl, "{ticker}", strings.ToUpper(strings.TrimSpace(ticker)))
	out = strings.ReplaceAll(out, "{h}", strconv.Itoa(h))
	return out
}

func getenv(k, def string) string {
	v := strings.TrimSpace(os.Getenv(k))
	if v == "" {
		return def
	}
	return v
}

func parsePredictionsCSV(r io.Reader) ([]map[string]any, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1

	header, err := cr.Read()
	if err != nil {
		return nil, err
	}
	for i := range header {
		header[i] = strings.TrimSpace(header[i])
	}

	var rows []map[string]any
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		row := map[string]any{}
		for i := 0; i < len(header) && i < len(rec); i++ {
			key := header[i]
			val := strings.TrimSpace(rec[i])

			if key == "horizon_days" || key == "pred_label" {
				if n, e := strconv.Atoi(val); e == nil {
					row[key] = n
					continue
				}
			}
			if key == "proba_up" || key == "threshold" {
				if f, e := strconv.ParseFloat(val, 64); e == nil {
					row[key] = f
					continue
				}
			}
			row[key] = val
		}

		rows = append(rows, row)
	}

	return rows, nil
}

func main() {
	cfg := mustConfig()

	minioClient, err := newMinio(cfg)
	if err != nil {
		panic(fmt.Errorf("minio client init error: %w", err))
	}

	r := gin.Default()

	r.GET("/health", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		exists, err := minioClient.BucketExists(ctx, cfg.PredBucket)
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "error",
				"minio":   "unreachable",
				"error":   err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":           "ok",
			"predictionsBucket": cfg.PredBucket,
			"bucketExists":     exists,
		})
	})

	// JSON
	r.GET("/predictions/:ticker", func(c *gin.Context) {
		ticker := c.Param("ticker")
		hStr := c.Query("h")
		if hStr == "" {
			hStr = "1"
		}
		h, err := strconv.Atoi(hStr)
		if err != nil || h <= 0 || h > 365 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "h must be a positive integer"})
			return
		}

		key := keyFromTemplate(cfg.KeyTemplate, ticker, h)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		obj, err := minioClient.GetObject(ctx, cfg.PredBucket, key, minio.GetObjectOptions{})
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "object not found", "bucket": cfg.PredBucket, "key": key})
			return
		}
		defer obj.Close()

		_, statErr := obj.Stat()
		if statErr != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "object not found", "bucket": cfg.PredBucket, "key": key})
			return
		}

		rows, err := parsePredictionsCSV(obj)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "csv parse error", "details": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"ticker": ticker,
			"h":      h,
			"bucket": cfg.PredBucket,
			"key":    key,
			"count":  len(rows),
			"data":   rows,
		})
	})

		// LIST OBJECTS (debug)
	r.GET("/objects", func(c *gin.Context) {
		prefix := c.Query("prefix")

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		var keys []string
		for obj := range minioClient.ListObjects(ctx, cfg.PredBucket, minio.ListObjectsOptions{
			Prefix:    prefix,
			Recursive: true,
		}) {
			if obj.Err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": obj.Err.Error()})
				return
			}
			keys = append(keys, obj.Key)
		}

		c.JSON(http.StatusOK, gin.H{
			"bucket": cfg.PredBucket,
			"prefix": prefix,
			"count":  len(keys),
			"keys":   keys,
		})
	})

		// FORECAST
	r.GET("/forecast/:ticker", func(c *gin.Context) {
		ticker := strings.ToUpper(strings.TrimSpace(c.Param("ticker")))
		hmaxStr := c.Query("hmax")
		if hmaxStr == "" {
			hmaxStr = "7"
		}
		hmax, err := strconv.Atoi(hmaxStr)
		if err != nil || hmax <= 0 || hmax > 30 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "hmax must be 1..30"})
			return
		}

		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		var out []gin.H
		for h := 1; h <= hmax; h++ {
			key := fmt.Sprintf("%s/H%d/forecast_latest.csv", ticker, h)

			obj, err := minioClient.GetObject(ctx, cfg.PredBucket, key, minio.GetObjectOptions{})
			if err != nil {
				out = append(out, gin.H{"h": h, "key": key, "error": "object not found"})
				continue
			}
			_, statErr := obj.Stat()
			if statErr != nil {
				_ = obj.Close()
				out = append(out, gin.H{"h": h, "key": key, "error": "object not found"})
				continue
			}

			rows, perr := parsePredictionsCSV(obj)
			_ = obj.Close()
			if perr != nil || len(rows) == 0 {
				out = append(out, gin.H{"h": h, "key": key, "error": "csv parse error"})
				continue
			}

			out = append(out, gin.H{"h": h, "key": key, "data": rows[0]})
		}

		c.JSON(http.StatusOK, gin.H{
			"ticker": ticker,
			"bucket": cfg.PredBucket,
			"hmax":   hmax,
			"items":  out,
		})
	})

	port := getenv("PORT", "8080")
	_ = r.Run("0.0.0.0:" + port)
}