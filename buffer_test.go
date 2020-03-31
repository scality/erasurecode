package erasurecode

import (
	"bytes"
	"io"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

type bufferTest struct {
	name                  string
	size, k, m, blockSize int
}

var bufferTests = []bufferTest{
	{
		name:      "200B",
		size:      200,
		k:         2,
		m:         1,
		blockSize: 32,
	},
	{
		name:      "262145B",
		size:      262145,
		k:         2,
		m:         1,
		blockSize: 32768,
	},
	{
		name:      "1MB",
		size:      1000000,
		k:         5,
		m:         1,
		blockSize: 32768,
	},
}

func TestBuffer(t *testing.T) {
	for _, test := range bufferTests {
		t.Run(test.name, func(t *testing.T) {
			size := test.size
			k := test.k
			blockSize := test.blockSize
			b := NewBufferMatrix(blockSize, size, k)
			data := make([]byte, size)
			for i := 0; i < size; i++ {
				data[i] = byte(i)
			}
			n, err := io.Copy(b, bytes.NewReader(data))

			assert.NoError(t, err)
			assert.Equal(t, int64(size), n)

			newData := b.RealData()
			assert.Equal(t, len(data), len(newData))
			assert.Equal(t, data, newData)

			b2 := NewBufferMatrix(blockSize, size, k)
			n, err = b2.ReadFrom(bytes.NewReader(data))

			assert.NoError(t, err)
			assert.Equal(t, int64(size), n)

			newData = b2.RealData()
			assert.Equal(t, len(data), len(newData))
			assert.Equal(t, data, newData)
		})
	}
}

func TestComparisonEncode(t *testing.T) {
	for _, test := range bufferTests {
		t.Run(test.name, func(t *testing.T) {
			size := test.size
			k := test.k
			m := test.m
			blockSize := test.blockSize
			b := NewBufferMatrix(blockSize, size, k)

			data := make([]byte, size)
			for i := 0; i < size; i++ {
				data[i] = byte(i)
			}
			n, err := io.Copy(b, bytes.NewReader(data))
			assert.Equal(t, size, int(n))
			b.Finish()
			assert.NoError(t, err)

			defer runtime.KeepAlive(b)

			backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: k, M: m, W: 8, HD: 5})
			defer backend.Close()

			encoded2, err := backend.EncodeMatrixWithBufferMatrix(b, blockSize)
			assert.NoError(t, err)
			defer encoded2.Free()

			encoded, err := backend.EncodeMatrix(data, blockSize)
			assert.NoError(t, err)
			defer encoded.Free()

			for j := 0; j < len(encoded2.Data); j++ {
				for i := 0; i < len(encoded2.Data[0]); i++ {
					assert.Equal(t, encoded2.Data[j][i], encoded.Data[j][i])
				}
			}
			for i := 0; i < k+m; i++ {
				assert.Equal(t, (encoded2.Data[i]), (encoded.Data[i]))
			}
		})
	}
}

func TestEncodeBufferMatrix(t *testing.T) {
	for _, test := range bufferTests {
		t.Run(test.name, func(t *testing.T) {
			size := test.size
			k := test.k
			m := test.m
			blockSize := test.blockSize
			b := NewBufferMatrix(blockSize, size, k)
			data := bytes.Repeat([]byte("A"), size)
			n, err := io.Copy(b, bytes.NewReader(data))
			assert.Equal(t, size, int(n))
			b.Finish()
			assert.NoError(t, err)

			defer runtime.KeepAlive(b)

			backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: k, M: m, W: 8, HD: 5})
			defer backend.Close()

			encoded, err := backend.EncodeMatrixWithBufferMatrix(b, blockSize)
			assert.NoError(t, err)
			defer encoded.Free()

			frags := encoded.Data

			for i := 0; i < k+m; i++ {
				assert.Equal(t, b.FragLen(), len(frags[i]))
			}

			decoded, err := backend.DecodeMatrix(frags, blockSize)
			assert.NoError(t, err)
			if err != nil {
				t.FailNow()
			}
			defer decoded.Free()

			assert.Equal(t, len(data), len(decoded.Data))
			assert.Equal(t, data, decoded.Data[:len(data)])

			for i := 0; i < len(data); i++ {
				assert.Equal(t, data[i], decoded.Data[i])
			}
		})
	}
}

// BenchmarkEncodeMatrix compares speeds of both style of encoding
// using a generic buffer (and then requiring some allocations in the C shim)
// using a specific buffer (less allocations)
func BenchmarkEncodeMatrix(b *testing.B) {
	for _, test := range bufferTests {
		b.Run(test.name, func(b *testing.B) {
			size := test.size
			k := test.k
			m := test.m
			blockSize := test.blockSize
			backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: k, M: m, W: 8, HD: 5})
			defer backend.Close()
			b.Run("original", func(b *testing.B) {
				buf := bytes.Repeat([]byte("A"), size)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					encoded, err := backend.EncodeMatrix(buf, blockSize)

					if err != nil {
						b.Fatal(err)
					}
					encoded.Free()
				}
			})

			b.Run("no copy", func(b *testing.B) {
				buf := NewBufferMatrix(blockSize, size, k)
				data := bytes.Repeat([]byte("A"), size)
				_, err := io.Copy(buf, bytes.NewReader(data))
				assert.NoError(b, err)
				buf.Finish()

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					encoded, err := backend.EncodeMatrixWithBufferMatrix(buf, blockSize)

					if err != nil {
						b.Fatal(err)
					}
					encoded.Free()
				}
			})
		})
	}
}

// BenchmarkBufferCopy compares basic buffer vs matrix implementation
// w.r.t. filling speeds. Please note the matrix implementation is expected
// to be slower. Speed gains will occur during encoding phase
func BenchmarkBufferCopy(b *testing.B) {
	for _, test := range bufferTests {
		b.Run(test.name, func(b *testing.B) {
			originalData := bytes.Repeat([]byte("A"), test.size)
			reader := bytes.NewReader(originalData)
			b.Run("original", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					reader.Seek(0, 0)
					buf := bytes.NewBuffer(make([]byte, 0, test.size))
					if _, err := io.Copy(buf, reader); err != nil {
						b.Fatal("cannot read buffer")
					}
				}
			})
			b.Run("buffermatrix", func(b *testing.B) {
				b.ResetTimer()
				buf := NewBufferMatrix(test.blockSize, test.size, test.k)
				for i := 0; i < b.N; i++ {
					reader.Seek(0, 0)
					buf.Reset(test.blockSize, test.size, test.k)
					if _, err := io.Copy(buf, reader); err != nil {
						b.Fatal("cannot read buffer")
					}
					buf.Finish()
				}
			})
		})
	}
}
