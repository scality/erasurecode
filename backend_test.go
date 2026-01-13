package erasurecode

import (
	"bytes"
	cryptorand "crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"reflect"
	"strings"
	"sync"
	"testing"
	"testing/quick"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var validParams = []Params{
	{Name: "isa_l_rs_vand", K: 2, M: 1},
	{Name: "isa_l_rs_vand", K: 2, M: 1, MaxBlockSize: 1},
	{Name: "isa_l_rs_vand", K: 2, M: 1, MaxBlockSize: maxBuffer * 2},
	{Name: "isa_l_rs_vand", K: 10, M: 4},
	{Name: "isa_l_rs_vand", K: 4, M: 3},
	{Name: "isa_l_rs_vand", K: 8, M: 4},
	{Name: "isa_l_rs_vand", K: 15, M: 4},
	{Name: "isa_l_rs_cauchy", K: 2, M: 1},
	{Name: "isa_l_rs_cauchy", K: 10, M: 4},
	{Name: "isa_l_rs_cauchy", K: 4, M: 3},
	{Name: "isa_l_rs_cauchy", K: 8, M: 4},
	{Name: "isa_l_rs_cauchy", K: 15, M: 4},
	{Name: "jerasure_rs_vand", K: 2, M: 1},
	{Name: "jerasure_rs_vand", K: 10, M: 4},
	{Name: "jerasure_rs_vand", K: 4, M: 3},
	{Name: "jerasure_rs_vand", K: 8, M: 4},
	{Name: "jerasure_rs_vand", K: 15, M: 4},
	{Name: "jerasure_rs_cauchy", K: 2, M: 1},
	{Name: "jerasure_rs_cauchy", K: 10, M: 4},
	{Name: "jerasure_rs_cauchy", K: 4, M: 3},
	{Name: "jerasure_rs_cauchy", K: 8, M: 4},
	{Name: "jerasure_rs_cauchy", K: 15, M: 4, W: 5},
}

var testPatterns = [][]byte{
	bytes.Repeat([]byte{0x00}, 1),
	bytes.Repeat([]byte{0xff}, 1),
	bytes.Repeat([]byte{0x00}, 1<<10),
	bytes.Repeat([]byte{0xff}, 1<<10),
	bytes.Repeat([]byte{0x00}, 1<<20),
	bytes.Repeat([]byte{0xff}, 1<<20),
	bytes.Repeat([]byte{0xf0, 0x0f}, 512),
	bytes.Repeat([]byte{0xde, 0xad, 0xbe, 0xef}, 256),
	bytes.Repeat([]byte{0xaa}, 1024),
	bytes.Repeat([]byte{0x55}, 1024),
	bytes.Repeat([]byte{0x55}, 2234345),
}

func shuf(src [][]byte) [][]byte {
	dest := make([][]byte, len(src))
	perm := rand.Perm(len(src))
	for i, v := range perm {
		dest[v] = src[i]
	}
	return dest
}

func TestGetVersion(t *testing.T) {
	v := GetVersion()
	t.Logf("INFO: Using liberasurecode version %s", v)
	if v.Major != 1 {
		t.Errorf("Expected major version number 1, not %d", v.Major)
	}
	if v.Less(Version{1, 4, 0}) {
		t.Errorf("liberasurecode_get_version was introduced in 1.4.0; got %v", v)
	}
}

func TestInitBackend(t *testing.T) {
	for _, params := range validParams {
		backend, err := InitBackend(params)
		if !BackendIsAvailable(params.Name) {
			if err == nil {
				t.Errorf("Expected EBACKENDNOTAVAIL")
			}
			continue
		}
		if err != nil {
			t.Errorf("%q", err)
			continue
		}
		if backend.libecDesc <= 0 {
			t.Errorf("Expected backend descriptor > 0, got %d", backend.libecDesc)
		}

		if err = backend.Close(); err != nil {
			t.Errorf("%q", err)
		}
		if err = backend.Close(); err == nil {
			t.Errorf("Expected error when closing an already-closed backend.")
		}
	}
}

func TestInitBackendFailure(t *testing.T) {
	cases := []struct {
		params Params
		want   string
	}{
		{Params{Name: "liberasurecode_rs_vand", K: -1, M: 1},
			"instance_create() returned EINVALIDPARAMS"},
		{Params{Name: "liberasurecode_rs_vand", K: 10, M: -1},
			"instance_create() returned EINVALIDPARAMS"},
		{Params{Name: "non-existent-backend", K: 10, M: 4},
			"unsupported backend \"non-existent-backend\""},
		{Params{Name: "", K: 10, M: 4},
			"unsupported backend \"\""},
		{Params{Name: "liberasurecode_rs_vand", K: 20, M: 20},
			"instance_create() returned EINVALIDPARAMS"},
		{Params{Name: "flat_xor_hd", K: 4, M: 4, HD: 3},
			"instance_create() returned EBACKENDINITERR"},
	}
	for _, args := range cases {
		backend, err := InitBackend(args.params)
		if err == nil {
			t.Errorf("Expected error when calling InitBackend(%v)",
				args.params)
			_ = backend.Close()
			continue
		}
		if err.Error() != args.want {
			t.Errorf("InitBackend(%v) produced error %q, want %q",
				args.params, err, args.want)
		}
		if backend.libecDesc != 0 {
			t.Errorf("InitBackend(%v) produced backend with descriptor %v, want 0",
				args.params, backend.libecDesc)
			_ = backend.Close()
		}
	}
}

func TestEncodeDecode(t *testing.T) {
	for _, params := range validParams {
		if !BackendIsAvailable(params.Name) {
			continue
		}
		backend, err := InitBackend(params)

		if err != nil {
			t.Errorf("Error creating backend %v: %q", params, err)
			continue
		}
		defer backend.Close()

		for patternIndex, pattern := range testPatterns {
			bm := NewBufferMatrix(DefaultChunkSize, len(pattern), backend.K)

			_, err = io.Copy(bm, bytes.NewReader(pattern))
			if err != nil {
				t.Errorf("Error copying pattern to buffer matrix: %q", err)
				break
			}
			bm.Finish()

			data, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
			require.NoError(t, err)
			defer data.Free()

			expectedVersion := GetVersion()
			frags := data.Data
			for index, frag := range frags {

				for i := range bm.SubGroups() {

					start := i * (DefaultChunkSize + 80)
					end := start + DefaultChunkSize + 80
					if i == bm.SubGroups()-1 {
						end = start + bm.FragLenLastSubGroup() + 80
					}

					piece := frag[start:end]
					require.True(t, backend.ValidateFragmentMatrix(piece, end-start-80))

					info := GetFragmentInfo(piece)
					if info.Index != index {
						t.Errorf("Expected frag %v to have index %v; got %v", index, index, info.Index)
					}
					if info.Size != len(piece)-80 { // 80 == sizeof (struct fragment_header_s)
						t.Errorf("Expected frag %v to have size %v; got %v", index, len(piece)-80, info.Size)
					}
					if info.BackendName != params.Name {
						t.Errorf("Expected frag %v to have backend %v; got %v", index, params.Name, info.BackendName)
					}
					if info.ErasureCodeVersion != expectedVersion {
						t.Errorf("Expected frag %v to have EC version %v; got %v", index, expectedVersion, info.ErasureCodeVersion)
					}
					if !info.IsValid {
						t.Errorf("Expected frag %v to be valid", index)
					}
				}
			}

			decode := func(frags [][]byte, description string) {
				decoded, err := backend.DecodeMatrix(frags, DefaultChunkSize)
				require.NoError(t, err)
				defer decoded.Free()
				require.True(t, bytes.Equal(decoded.Data, pattern),
					"%v:%d(%v) pattern: %v, got: %q",
					description, patternIndex, backend, pattern, decoded.Data)
			}

			decode(frags, "all frags")
			decode(shuf(frags), "all frags, shuffled")
			decode(frags[:params.K], "data frags")
			decode(shuf(frags[:params.K]), "shuffled data frags")
			decode(frags[params.M:], "with parity frags")
			decode(shuf(frags[params.M:]), "shuffled parity frags")

		}
	}
}

func TestReconstruct(t *testing.T) {
	for _, params := range validParams {
		if !BackendIsAvailable(params.Name) {
			continue
		}
		backend, err := InitBackend(params)
		if err != nil {
			t.Errorf("Error creating backend %v: %q", params, err)
			_ = backend.Close()
			continue
		}
		defer func() {
			_ = backend.Close()
		}()

		for patternIndex, pattern := range testPatterns {

			t.Run(fmt.Sprintf("%s_%d_%d-%d-%d",
				params.Name, params.K, params.M,
				patternIndex,
				len(pattern)),
				func(t *testing.T) {
					bm := NewBufferMatrix(DefaultChunkSize, len(pattern), backend.K)
					_, err = io.Copy(bm, bytes.NewReader(pattern))
					require.NoError(t, err)
					bm.Finish()
					data, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
					require.NoError(t, err)
					defer data.Free()
					frags := data.Data

					reconstruct := func(recon_frags [][]byte, frag_index int, description string) {
						data, err := backend.ReconstructMatrix(recon_frags, frag_index, DefaultChunkSize)
						require.NoError(t, err, "%v: %v: %q for pattern %d", description, backend, err, patternIndex)
						defer data.Free()
						require.True(t, bytes.Equal(data.Data, frags[frag_index]), "%v: Expected %v to roundtrip pattern %d, got %q", description, backend, patternIndex, data.Data)
					}
					reconstruct(shuf(frags[:params.K]), params.K+params.M-1, "last frag from data frags")
					reconstruct(shuf(frags[params.M:]), 0, "first frag with parity frags")
				})
		}
	}
}

func TestIsInvalidFragment(t *testing.T) {
	for _, params := range validParams {
		if !BackendIsAvailable(params.Name) {
			continue
		}
		backend, err := InitBackend(params)
		if err != nil {
			t.Errorf("Error creating backend %v: %q", params, err)
			_ = backend.Close()
			continue
		}
		defer func() {
			_ = backend.Close()
		}()
		for patternIndex, pattern := range testPatterns {
			bm := NewBufferMatrix(DefaultChunkSize, len(pattern), backend.K)
			_, err = io.Copy(bm, bytes.NewReader(pattern))
			require.NoError(t, err)
			bm.Finish()
			data, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
			require.NoError(t, err)
			defer data.Free()

			parts := data.Data
			for index, part := range parts {
				for i := range bm.SubGroups() {

					start := i * (DefaultChunkSize + 80)
					end := start + DefaultChunkSize + 80
					if i == bm.SubGroups()-1 {
						end = start + bm.FragLenLastSubGroup() + 80
					}

					frag := part[start:end]

					if backend.IsInvalidFragment(frag) {
						t.Errorf("%v: frag %v unexpectedly invalid for pattern %d", backend, index, patternIndex)
					}
					fragCopy := make([]byte, len(frag))
					copy(fragCopy, frag)

					// corrupt the frag
					corruptedByte := rand.Intn(len(frag)) //nolint:gosec
					for 71 <= corruptedByte && corruptedByte < 80 {
						// in the alignment padding -- try again
						corruptedByte = rand.Intn(len(frag)) //nolint:gosec
					}
					frag[corruptedByte] ^= 0xff
					if !backend.IsInvalidFragment(frag) {
						t.Errorf("%v: frag %v unexpectedly valid after inverting byte %d for pattern %d", backend, index, corruptedByte, patternIndex)
					}
					if corruptedByte < 4 || 8 <= corruptedByte && corruptedByte <= 59 {
						/** corruption is in metadata; claim we were created by a version of
						 *  libec that predates metadata checksums. Note that
						 *  Note that a corrupted fragment size (bytes 4-7) will lead to a
						 *  segfault when we try to verify the fragment -- there's a reason
						 *  we added metadata checksums!
						 */
						copy(frag[63:67], []byte{9, 1, 1, 0})
						if 20 <= corruptedByte && corruptedByte <= 53 {
							/** Corrupted data checksum type or data checksum
							 *  We may or may not detect this type of error; in particular,
							 *      - if data checksum type is not in ec_checksum_type_t,
							 *        it is ignored
							 *      - if data checksum is mangled, we may still be valid
							 *        under the "alternative" CRC32; this seems more likely
							 *        with the byte inversion when the data is short
							 *  Either way, though, clearing the checksum type should make
							 *  it pass.
							 */
							frag[20] = 0
							if backend.IsInvalidFragment(frag) {
								t.Errorf("%v: frag %v unexpectedly invalid after clearing metadata crc and disabling data crc", backend, index)
							}
						} else if corruptedByte >= 54 || 0 <= corruptedByte && corruptedByte < 4 {
							/** Some corruptions of some bytes are still detectable. Since we're
							 *  inverting the byte, we can detect:
							 *      - frag index -- bytes 0-3
							 *      - data checksum type -- byte 20
							 *      - data checksum mismatch -- byte 54
							 *      - backend id -- byte 55
							 *      - backend version -- bytes 56-59
							 */
							if !backend.IsInvalidFragment(frag) {
								t.Errorf("%v: frag %v unexpectedly still valid after clearing metadata crc", backend, index)
							}
						} else {
							if backend.IsInvalidFragment(frag) {
								t.Errorf("%v: frag %v unexpectedly invalid after clearing metadata crc", backend, index)
							}
						}
					} else if corruptedByte >= 67 {
						copy(frag[20:25], []byte{1, 0, 0, 0, 0})
						// And since we've changed the metadata, roll back version as above...
						copy(frag[63:67], []byte{9, 1, 1, 0})
						if backend.IsInvalidFragment(frag) {
							t.Errorf("%v: frag %v unexpectedly invalid after clearing data crc", backend, index)
							t.FailNow()
						}
					}
					frag[corruptedByte] ^= 0xff
					copy(frag[63:67], fragCopy[63:67])
					copy(frag[20:25], fragCopy[20:25])

					if !bytes.Equal(frag, fragCopy) {
						for i, orig := range fragCopy {
							if frag[i] != orig {
								t.Logf("%v != %v at index %v", frag[i], orig, i)
							}
						}
						t.Fatal(corruptedByte, frag, fragCopy)
					}

					frag[corruptedByte]++
					if !backend.IsInvalidFragment(frag) {
						t.Errorf("%v: frag %v unexpectedly valid after incrementing byte %d for pattern %d", backend, index, corruptedByte, patternIndex)
					}
					frag[corruptedByte] -= 2
					if corruptedByte >= 63 && corruptedByte < 67 && frag[corruptedByte] != 0xff {
						if backend.IsInvalidFragment(frag) {
							t.Errorf("%v: frag %v unexpectedly invalid after decrementing version byte %d for pattern %d", backend, index, corruptedByte, patternIndex)
						}
					} else {
						if !backend.IsInvalidFragment(frag) {
							t.Errorf("%v: frag %v unexpectedly valid after decrementing byte %d for pattern %d", backend, index, corruptedByte, patternIndex)
						}
					}
				}
			}
		}
	}
}

func TestBackendIsAvailable(t *testing.T) {
	requiredBackends := []string{
		"null",
		"flat_xor_hd",
		"liberasurecode_rs_vand",
	}
	optionalBackends := []string{
		"isa_l_rs_vand",
		"isa_l_rs_cauchy",
		"jerasure_rs_vand",
		"jerasure_rs_cauchy",
		"shss",
		"libphazr",
	}
	for _, name := range requiredBackends {
		if !BackendIsAvailable(name) {
			t.Fatalf("%v is not available", name)
		}
	}
	for _, name := range optionalBackends {
		if !BackendIsAvailable(name) {
			t.Logf("INFO: backend not available: %v", name)
		}
	}
}

// TestGC do multiple decode / reconstruct in concurrent goroutine.
// When enough data has been allocated and freed, the GC will wakeup
// and may destroy a block in-use. If this block is re-allocated,
// the content is most likely no longer valid and it should
// trigger an error or a crash.
func TestGC(t *testing.T) {
	input := bytes.Repeat([]byte("X"), 1000000)
	backend, err := InitBackend(
		Params{
			Name: "isa_l_rs_vand",
			K:    2,
			M:    1,
		})

	if err != nil {
		t.Logf("Cannot run test because %s", err)
		return
	}

	tests := []struct {
		name     string
		testFunc func()
	}{{
		"Reconstruct",
		func() {
			bm := NewBufferMatrix(DefaultChunkSize, len(input), backend.K)
			_, err = io.Copy(bm, bytes.NewReader(input))
			require.NoError(t, err)
			bm.Finish()
			encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
			require.NoError(t, err)
			defer encoded.Free()

			vect := encoded.Data

			oldData := vect[0][:] // force a copy

			data, err := backend.ReconstructMatrix(vect[1:3], 0, DefaultChunkSize)
			require.NoError(t, err)
			defer data.Free()
			require.True(t, bytes.Equal(data.Data, oldData))
		},
	},
		{
			"Decode",
			func() {
				bm := NewBufferMatrix(DefaultChunkSize, len(input), backend.K)
				_, err = io.Copy(bm, bytes.NewReader(input))
				require.NoError(t, err)
				bm.Finish()
				encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
				require.NoError(t, err)
				defer encoded.Free()

				vect := encoded.Data

				decoded, err := backend.DecodeMatrix(vect[0:2], DefaultChunkSize)
				require.NoError(t, err)
				defer decoded.Free()
				require.True(t, bytes.Equal(decoded.Data, input))
			},
		},
	}

	nbRoutines := 500

	for _, test := range tests {
		t.Run(test.name, func(_ *testing.T) {
			var wg sync.WaitGroup
			wg.Add(nbRoutines)

			for i := 0; i < nbRoutines; i++ {
				go func() {
					test.testFunc()
					wg.Done()
				}()
			}
			wg.Wait()
		})
	}
	_ = backend.Close()
}

func TestAvailableBackends(t *testing.T) {
	for _, name := range AvailableBackends() {
		backend, err := InitBackend(Params{Name: name, K: 3, M: 3, HD: 3})
		if err != nil {
			t.Errorf("Error creating backend %v: %q", name, err)
		}
		_ = backend.Close()
	}
	t.Logf("INFO: found %v/%v available backends", len(AvailableBackends()), len(KnownBackends))
}

func BenchmarkEncode(b *testing.B) {
	backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})

	buf := bytes.Repeat([]byte("A"), 1024*1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm := NewBufferMatrix(DefaultChunkSize, len(buf), backend.K)
		_, err := io.Copy(bm, bytes.NewReader(buf))
		require.NoError(b, err)
		bm.Finish()
		encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
		require.NoError(b, err)
		defer encoded.Free()
	}
	_ = backend.Close()
}

const DefaultChunkSize = 32768
const DefaultFragSize = 1048576

type decodeTest struct {
	size int
	p    Params
}

func (d decodeTest) String() string {
	return fmt.Sprintf("%s-%d+%d-%db", d.p.Name, d.p.K, d.p.M, d.size)
}

var decodeTests = []decodeTest{
	{1000 * 1000, Params{Name: "isa_l_rs_vand", K: 5, M: 1}},
	{1024 * 1024, Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5}},
	{5 * 100000, Params{Name: "isa_l_rs_vand", K: 5, M: 7}},
	// Will force an allocation of a new (dedicated) pool
	{10000000, Params{Name: "isa_l_rs_vand", K: 2, M: 1, MaxBlockSize: 10000000 + maxBuffer}},
	// Will force an allocation at every encoding (but should work)
	{maxBuffer * 2, Params{Name: "isa_l_rs_vand", K: 2, M: 1, MaxBlockSize: maxBuffer / 2}},
}

func BenchmarkLinearizeM(b *testing.B) {
	for _, test := range decodeTests {
		b.Run(test.String(), func(b *testing.B) {
			backend, err := InitBackend(test.p)
			if err != nil {
				b.Fatal("cannot create backend", err)
			}
			defer func() {
				_ = backend.Close()
			}()

			buf := bytes.Repeat([]byte("A"), test.size)
			bm := NewBufferMatrix(DefaultChunkSize, len(buf), backend.K)
			_, err = io.Copy(bm, bytes.NewReader(buf))
			require.NoError(b, err)
			bm.Finish()
			encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
			require.NoError(b, err)
			defer encoded.Free()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				decoded, err := backend.LinearizeMatrix(encoded.Data, DefaultChunkSize)
				require.NoError(b, err)
				defer decoded.Free()
				require.True(b, bytes.Equal(decoded.Data, buf))
			}
		})
	}
}

func BenchmarkDecodeM(b *testing.B) {
	for _, test := range decodeTests {
		b.Run(test.String(), func(b *testing.B) {
			backend, err := InitBackend(test.p)
			if err != nil {
				b.Fatal("cannot create backend", err)
			}
			defer func() {
				_ = backend.Close()
			}()

			buf := bytes.Repeat([]byte("A"), test.size)
			bm := NewBufferMatrix(DefaultChunkSize, len(buf), backend.K)
			_, err = io.Copy(bm, bytes.NewReader(buf))
			require.NoError(b, err)
			bm.Finish()
			encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
			require.NoError(b, err)

			defer encoded.Free()

			data := encoded.Data[1:]
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				decoded, err := backend.DecodeMatrix(data, DefaultChunkSize)
				require.NoError(b, err)
				defer decoded.Free()
				require.True(b, bytes.Equal(decoded.Data, buf))
			}
		})
	}
}

func BenchmarkReconstruct(b *testing.B) {
	for _, test := range decodeTests {
		b.Run(test.String(), func(b *testing.B) {
			backend, err := InitBackend(test.p)
			if err != nil {
				b.Fatal("cannot create backend", err)
			}
			defer func() {
				_ = backend.Close()
			}()

			buf := bytes.Repeat([]byte("A"), test.size)
			bm := NewBufferMatrix(DefaultChunkSize, len(buf), backend.K)
			_, err = io.Copy(bm, bytes.NewReader(buf))
			require.NoError(b, err)
			bm.Finish()
			encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
			require.NoError(b, err)
			defer encoded.Free()
			flags := encoded.Data[1:]
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				data, err := backend.ReconstructMatrix(flags, 0, DefaultChunkSize)
				require.NoError(b, err, "cannot reconstruct matrix: %v", err)
				defer data.Free()
				require.True(b, bytes.Equal(data.Data, encoded.Data[0]))
			}
		})
	}
}

func BenchmarkMatrix(b *testing.B) {
	for _, dtest := range decodeTests {
		blockSize := 32768
		// If you want 1 single block per fragment use:
		// blockSize := (dtest.size + dtest.p.K - 1) / dtest.p.K
		buf := bytes.Repeat([]byte("X"), dtest.size)
		b.Run(fmt.Sprintf("%s:%d+%d,size=%d", dtest.p.Name, dtest.p.K, dtest.p.M, dtest.size),
			func(b *testing.B) {
				for _, crc := range []int{ /* ChecksumNone, */ ChecksumCrc32, ChecksumXxhash} {
					b.Run(fmt.Sprintf("crc=%v", crc),
						func(b *testing.B) {
							dtest.p.Checksum = crc
							backend, _ := InitBackend(dtest.p)
							defer func() {
								_ = backend.Close()
							}()
							b.Run("Encode", func(b *testing.B) {
								b.ResetTimer()
								for i := 0; i < b.N; i++ {
									bm := NewBufferMatrix(DefaultChunkSize, len(buf), backend.K)
									_, err := io.Copy(bm, bytes.NewReader(buf))
									require.NoError(b, err)
									bm.Finish()
									encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
									require.NoError(b, err)
									defer encoded.Free()
								}
							})
							b.Run("Decode", func(b *testing.B) {
								bm := NewBufferMatrix(DefaultChunkSize, len(buf), backend.K)
								_, err := io.Copy(bm, bytes.NewReader(buf))
								require.NoError(b, err)
								bm.Finish()
								encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
								require.NoError(b, err)
								defer encoded.Free()

								b.ResetTimer()
								for i := 0; i < b.N; i++ {
									decoded, err := backend.LinearizeMatrix(encoded.Data, blockSize)
									if err != nil {
										b.Fatal(err)
									}
									decoded.Free()
								}
							})
							b.Run("Reconstruct", func(b *testing.B) {
								bm := NewBufferMatrix(DefaultChunkSize, len(buf), backend.K)
								_, err := io.Copy(bm, bytes.NewReader(buf))
								require.NoError(b, err)
								bm.Finish()
								encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
								require.NoError(b, err)
								defer encoded.Free()
								b.ResetTimer()
								for i := 0; i < b.N; i++ {

									decoded, err := backend.ReconstructMatrix(encoded.Data[1:], 0, blockSize)
									require.NoError(b, err)
									defer decoded.Free()
								}
							})
						})
				}
			})
	}
}

func BenchmarkDecode(b *testing.B) {
	backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})
	defer func() {
		_ = backend.Close()
	}()
	buf := bytes.Repeat([]byte("A"), 1024*1024)
	bm := NewBufferMatrix(DefaultChunkSize, len(buf), backend.K)
	_, err := io.Copy(bm, bytes.NewReader(buf))
	require.NoError(b, err)
	bm.Finish()
	res, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
	require.NoError(b, err)

	defer res.Free()

	for i := 0; i < b.N; i++ {
		decoded, _ := backend.DecodeMatrix(res.Data, DefaultChunkSize)
		defer decoded.Free()

	}

}

func TestEncodeM(t *testing.T) {
	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})

	if err != nil {
		t.Fatalf("cannot init backend: (%v)", err)
	}

	buf := make([]byte, 1024*1024)
	_, _ = cryptorand.Read(buf)

	testParams := []struct {
		chunkUnit   int
		lenToDecode int
	}{
		{chunkUnit: 4096, lenToDecode: 4097},
		{chunkUnit: 4097, lenToDecode: 4096},
		{chunkUnit: 4096, lenToDecode: len(buf)},
		{chunkUnit: 4096, lenToDecode: 4096},
	}

	for _, param := range testParams {
		p := param
		testName := fmt.Sprintf("TestEncodeB-%d-%d", p.chunkUnit, p.lenToDecode)
		t.Run(testName, func(t *testing.T) {
			// Do the matrix encoding
			bm := NewBufferMatrix(p.chunkUnit, len(buf), backend.K)
			_, err := io.Copy(bm, bytes.NewReader(buf))
			require.NoError(t, err)
			bm.Finish()
			result, err := backend.EncodeMatrixWithBufferMatrix(bm, p.chunkUnit)
			require.NoError(t, err)
			defer result.Free()

			// Check that our linearized buffer
			// contains expected data when there is all data fragment.
			ddata, err := backend.LinearizeMatrix(result.Data, p.chunkUnit)
			require.NoError(t, err)
			defer ddata.Free()
			require.Equal(t, len(buf), len(ddata.Data), "data mismatch")
			require.True(t, bytes.Equal(buf, ddata.Data), "data mismatch")

			/* now do the same but with the slow path*/
			/* we will run a matrix decoding but withtout some data part, to enforce repairing*/
			var vect [][]byte
			vect = append(vect, result.Data[2])
			vect = append(vect, result.Data[3])
			vect = append(vect, result.Data[4])
			vect = append(vect, result.Data[5])

			ddata2, _ := backend.DecodeMatrix(vect, p.chunkUnit)
			require.True(t, bytes.Equal(buf, ddata2.Data), "data mismatch")
			defer ddata2.Free()
		})
	}
	_ = backend.Close()
}

func TestLinearizeMatrixAndReconstruct(t *testing.T) {
	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: 2, M: 1, W: 8, HD: 5})
	require.NoError(t, err)
	defer func() {
		_ = backend.Close()
	}()

	testParams := []struct {
		chunkSize    int
		dataSize     int
		startIncl    int
		endIncl      int
		useOldFormat bool
	}{
		{
			chunkSize:    512,
			dataSize:     512*2 + 10,
			startIncl:    512*2 - 3,
			endIncl:      512*2 - 1,
			useOldFormat: false,
		},
		{
			chunkSize:    512,
			dataSize:     512*2 + 10,
			startIncl:    512*2 - 3,
			endIncl:      512*2 - 1,
			useOldFormat: true,
		},
		{
			chunkSize:    DefaultChunkSize,
			dataSize:     105623,
			startIncl:    59441,
			endIncl:      64149,
			useOldFormat: false,
		},

		{
			chunkSize:    DefaultChunkSize,
			dataSize:     105623,
			startIncl:    59441,
			endIncl:      64149,
			useOldFormat: true,
		},
		{
			chunkSize:    DefaultChunkSize,
			dataSize:     105623,
			startIncl:    105610,
			endIncl:      105622,
			useOldFormat: true,
		},
		{
			chunkSize:    DefaultChunkSize,
			dataSize:     105623,
			startIncl:    105610,
			endIncl:      105622,
			useOldFormat: false,
		},
	}

	for _, param := range testParams {
		p := param
		testName := fmt.Sprintf("TestLinearizeMatrixAndReconstruct(oldformat=%v)-%d-%d-%d-%d",
			p.useOldFormat, p.chunkSize, p.dataSize, p.startIncl, p.endIncl,
		)
		t.Run(testName, func(t *testing.T) {
			currentChunkSize := p.chunkSize
			dataSize := p.dataSize
			startIncl := p.startIncl
			endIncl := p.endIncl

			data := make([]byte, dataSize)
			for i := range dataSize {
				data[i] = byte('A' + i%26)
			}
			bm := NewBufferMatrix(currentChunkSize, len(data), backend.K)
			if p.useOldFormat {
				bm.UseOldFormat()
			}
			_, err = io.Copy(bm, bytes.NewReader(data))
			require.NoError(t, err)
			bm.Finish()
			encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, currentChunkSize)
			require.NoError(t, err)
			defer encoded.Free()

			rangeM := backend.GetRangeMatrix(startIncl, endIncl, currentChunkSize, len(encoded.Data[0]))
			require.NotNil(t, rangeM)

			/* Decode the matrix as if it was requested and
			   checks that the result matches the payload on the requested range. */
			frags := make([][]byte, 0)
			for i := 0; i < rangeM.FragCount; i++ {
				fragIdx := (rangeM.FragFirstIncl + i) % backend.K
				buffer := encoded.Data[fragIdx][rangeM.InFragRangeStartIncl:rangeM.InFragRangeEndExcl]
				frags = append(frags, buffer)
			}

			decoded, err := backend.LinearizeMatrix(frags, currentChunkSize)
			require.NoError(t, err)
			defer decoded.Free()

			expected := data[startIncl:endIncl]

			linearizedRangeEndExcl := rangeM.LinearizedRangeStartIncl + (endIncl - startIncl)
			found := decoded.Data[rangeM.LinearizedRangeStartIncl:linearizedRangeEndExcl]

			require.True(t, bytes.Equal(expected, found))

			frags2 := make([][]byte, 0)
			for i := 0; i < backend.K+backend.M; i++ {
				frags2 = append(frags2, encoded.Data[i][rangeM.InFragRangeStartIncl:rangeM.InFragRangeEndExcl])
			}

			// now do the same test, but this time, instead of linearizing, we are going to reconstruct the stripes
			reconstructed, err := backend.ReconstructMatrix(frags2[1:], 0, currentChunkSize)

			require.NoError(t, err)
			defer reconstructed.Free()

			require.True(t, bytes.Equal(frags2[0], reconstructed.Data))
		})
	}
}

func TestLinearizeMatrix(t *testing.T) {
	pieceSize := DefaultChunkSize
	k := 4
	m := 1

	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: k, M: m, W: 8, HD: m})
	require.NoError(t, err)
	defer func() {
		_ = backend.Close()
	}()

	rangeValues := func(values []reflect.Value, rng *rand.Rand) {
		dataSize := 1 + rng.Intn(7*1024*1024)

		/* To avoid generating too many unintersting samples, this tests
		   focuses on valid inputs. */
		startIncl := rng.Intn(dataSize - 1)
		endIncl := rng.Intn(dataSize - 1)
		if endIncl < startIncl {
			tmp := startIncl
			startIncl = endIncl
			endIncl = tmp
		}

		values[0] = reflect.ValueOf(startIncl)
		values[1] = reflect.ValueOf(endIncl)
		values[2] = reflect.ValueOf(dataSize)
	}

	checkRange := func(startIncl, endIncl, dataSize int) bool {
		t.Logf("TestLinearizeMatrix check %d-%d-%d", startIncl, endIncl, dataSize)

		data := make([]byte, dataSize)
		_, _ = cryptorand.Read(data)

		bm := NewBufferMatrix(DefaultChunkSize, len(data), backend.K)
		_, err := io.Copy(bm, bytes.NewReader(data))
		require.NoError(t, err)
		bm.Finish()
		encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
		require.NoError(t, err)
		defer encoded.Free()

		fragSize := len(encoded.Data[0])
		rangeM := backend.GetRangeMatrix(startIncl, endIncl, pieceSize, fragSize)
		require.NotNil(t, rangeM)

		/* Decode the matrix as if it was requested and
		   checks that the result matches the payload on the requested range. */
		frags := make([][]byte, 0)
		for i := 0; i < rangeM.FragCount; i++ {
			fragIdx := (rangeM.FragFirstIncl + i) % k
			buffer := encoded.Data[fragIdx][rangeM.InFragRangeStartIncl:rangeM.InFragRangeEndExcl]
			frags = append(frags, buffer)
		}

		decoded, err := backend.LinearizeMatrix(frags, pieceSize)
		require.NoError(t, err)
		defer decoded.Free()

		expected := data[startIncl : endIncl+1]

		linearizedRangeEndExcl := rangeM.LinearizedRangeStartIncl + (endIncl - startIncl) + 1
		found := decoded.Data[rangeM.LinearizedRangeStartIncl:linearizedRangeEndExcl]
		return bytes.Equal(expected, found)
	}

	config := quick.Config{
		Values: rangeValues,
	}

	require.NoError(t, quick.Check(checkRange, &config))

}

func TestDecodeMatrix(t *testing.T) {
	assert := assert.New(t)

	pieceSize := DefaultChunkSize
	k := 4
	m := 1

	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: k, M: m, W: 8, HD: m})
	if err != nil {
		t.Fatalf("cannot init backend: (%v)", err)
	}

	rangeValues := func(values []reflect.Value, rng *rand.Rand) {
		dataSize := 1 + rng.Intn(7*1024*1024)

		startIncl := rng.Intn(dataSize - 1)
		endIncl := rng.Intn(dataSize - 1)
		if endIncl < startIncl {
			tmp := startIncl
			startIncl = endIncl
			endIncl = tmp
		}

		failedFragIdx := rng.Intn(k + m)

		values[0] = reflect.ValueOf(startIncl)
		values[1] = reflect.ValueOf(endIncl)
		values[2] = reflect.ValueOf(dataSize)
		values[3] = reflect.ValueOf(failedFragIdx)
	}

	checkRange := func(startIncl, endIncl, dataSize int, failedFragIdx int) bool {
		t.Logf("TestDecodeMatrix check %d-%d-%d-%d", startIncl, endIncl, dataSize, failedFragIdx)

		data := make([]byte, dataSize)
		_, _ = cryptorand.Read(data)

		bm := NewBufferMatrix(DefaultChunkSize, len(data), backend.K)
		_, err := io.Copy(bm, bytes.NewReader(data))
		require.NoError(t, err)
		bm.Finish()
		encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
		require.NoError(t, err)
		defer encoded.Free()

		fragSize := len(encoded.Data[0])
		rangeM := backend.GetRangeMatrix(startIncl, endIncl, pieceSize, fragSize)
		assert.NotNil(rangeM)

		/* Decode the matrix as if it was requested and
		   checks that the result matches the payload on the requested range. */
		frags := make([][]byte, 0)
		for i := 0; i < (k + m); i++ {
			fragIdx := i
			if fragIdx == failedFragIdx {
				continue
			}

			buffer := encoded.Data[fragIdx][rangeM.InFragRangeStartIncl:rangeM.InFragRangeEndExcl]
			frags = append(frags, buffer)
		}

		decoded, err := backend.DecodeMatrix(frags, pieceSize)
		assert.Nil(err)
		defer decoded.Free()

		expected := data[startIncl : endIncl+1]

		decodedRangeEndExcl := rangeM.DecodedRangeStartIncl + (endIncl - startIncl) + 1
		found := decoded.Data[rangeM.DecodedRangeStartIncl:decodedRangeEndExcl]
		return bytes.Equal(expected, found)
	}

	config := quick.Config{
		Values: rangeValues,
	}

	if err := quick.Check(checkRange, &config); err != nil {
		t.Error(err)
	}
}

func TestValidateFragmentMatrix(t *testing.T) {
	assert := assert.New(t)

	pieceSize := DefaultChunkSize
	k := 4
	m := 1

	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: k, M: m, W: 8, HD: m})
	if err != nil {
		t.Fatalf("cannot init backend: (%v)", err)
	}

	dataSize := 7 * 1024 * 1024
	data := make([]byte, dataSize)
	_, _ = cryptorand.Read(data)

	bm := NewBufferMatrix(DefaultChunkSize, len(data), backend.K)
	_, err = io.Copy(bm, bytes.NewReader(data))
	require.NoError(t, err)
	bm.Finish()
	encoded, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
	require.NoError(t, err)
	defer encoded.Free()

	fragSize := len(encoded.Data[0])
	for i := 0; i < len(encoded.Data); i++ {
		rangeMatrix := backend.GetRangeMatrix(0, dataSize-1, pieceSize, fragSize)
		assert.NotNil(rangeMatrix)

		frag := encoded.Data[i][rangeMatrix.InFragRangeStartIncl:rangeMatrix.InFragRangeEndExcl]
		valid := backend.ValidateFragmentMatrix(frag, pieceSize)
		assert.True(valid)

		chunkSize := pieceSize + backend.headerSize
		offset := 0
		for offset < len(frag) {
			for altered := 0; altered < backend.headerSize; altered++ {
				t.Logf("frag %d altered offset %d altered %d", i, offset, altered)
				previous := frag[offset+altered]
				frag[offset+altered] = previous + 1

				valid := backend.ValidateFragmentMatrix(frag, pieceSize)

				/* libec_version and padding not checked */
				if altered >= 63 && altered < 67 {
					assert.True(valid)
				} else if altered >= 71 {
					assert.True(valid)
				} else {
					assert.False(valid)
				}

				frag[offset+altered] = previous
			}

			offset += chunkSize
		}
	}
}

func TestReconstructM(t *testing.T) {
	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})

	if err != nil {
		t.Fatalf("cannot init backend: (%v)", err)
	}

	buf := make([]byte, 1024*1024)
	for i := 0; i < len(buf); i++ {
		buf[i] = byte('A' + i%26)
	}

	// All our sub tests case. Each {X,Y} represents respectively the chunking unit (size of each subpart)
	// and the fragment number we want to have to reconstruct
	testParams := []struct {
		chunkUnit  int
		fragNumber int
	}{
		{chunkUnit: 4096, fragNumber: 0},
		{chunkUnit: 4096, fragNumber: backend.K},
		{chunkUnit: DefaultChunkSize, fragNumber: 1},
	}

	for _, param := range testParams {
		p := param
		testName := fmt.Sprintf("TestReconstruct-%d-%d", p.chunkUnit, p.fragNumber)
		t.Run(testName, func(t *testing.T) {
			// Do the matrix encoding
			bm := NewBufferMatrix(p.chunkUnit, len(buf), backend.K)
			_, err := io.Copy(bm, bytes.NewReader(buf))
			require.NoError(t, err)
			bm.Finish()
			result, err := backend.EncodeMatrixWithBufferMatrix(bm, p.chunkUnit)
			require.NoError(t, err)

			defer result.Free()

			var vect [][]byte
			for i := 0; i < backend.K+backend.M; i++ {
				if i != p.fragNumber {
					vect = append(vect, result.Data[i])
				}
			}

			ddata, err := backend.ReconstructMatrix(vect, p.fragNumber, p.chunkUnit)
			require.NoError(t, err)
			require.NotNil(t, ddata)

			res := bytes.Compare(ddata.Data, result.Data[p.fragNumber])
			require.Equal(t, 0, res)
			defer ddata.Free()
		})
	}
	_ = backend.Close()
}

func TestEncodeDecodeMatrix(t *testing.T) {
	for _, params := range validParams {
		if strings.Contains(params.Name, "jerasure_rs_cauchy") {
			t.Logf("Skipping %s, not working with matrix", params.Name)
			continue
		}
		if !BackendIsAvailable(params.Name) {
			continue
		}
		backend, err := InitBackend(params)
		if err != nil {
			t.Errorf("Error creating backend %v: %q", params, err)
			continue
		}
		defer func() {
			_ = backend.Close()
		}()
		for patternIndex, pattern := range testPatterns {
			t.Run(fmt.Sprintf("%s_%d_%d-%d-%d",
				params.Name, params.K, params.M,
				patternIndex,
				len(pattern)),
				func(t *testing.T) {
					bm := NewBufferMatrix(DefaultChunkSize, len(pattern), backend.K)
					_, err := io.Copy(bm, bytes.NewReader(pattern))
					require.NoError(t, err)
					bm.Finish()
					data, err := backend.EncodeMatrixWithBufferMatrix(bm, DefaultChunkSize)
					require.NoError(t, err)
					defer data.Free()

					frags := data.Data
					decode := func(frags [][]byte, description string) {
						decoded, err := backend.DecodeMatrix(frags, DefaultChunkSize)
						require.NoError(t, err)
						require.True(t, bytes.Equal(decoded.Data, pattern), "%v: Expected %v to roundtrip pattern %d, got %q", description, backend, patternIndex, decoded.Data)
						defer decoded.Free()
					}

					decode(frags, "all frags")
					decode(shuf(frags), "all frags, shuffled")
					decode(frags[:params.K], "data frags")
					decode(shuf(frags[:params.K]), "shuffled data frags")
					decode(frags[params.M:], "with parity frags")
					decode(shuf(frags[params.M:]), "shuffled parity frags")

					for fIdx := 0; fIdx < params.K; fIdx++ {
						newFrags := frags[fIdx+1:]
						if fIdx >= 1 {
							newFrags = append(newFrags, frags[0:fIdx]...)
						}
						ddata, err := backend.ReconstructMatrix(newFrags, fIdx, DefaultChunkSize)
						require.NoError(t, err)
						require.True(t, bytes.Equal(ddata.Data, frags[fIdx]), "part %d reconstructed not equal to original len: %q != %q", fIdx, ddata.Data, frags[fIdx])
						defer ddata.Free()
					}

				})
		}
	}
}

func TestRangeMatrix(t *testing.T) {
	// These are basic tests on the rangeMatrix. More complete tests on the
	// actual values are performed by the random tests:
	// TestDecodeMatrix and TestLinearizeMatrix.
	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})
	if err != nil {
		t.Fatalf("cannot init backend: (%v)", err)
	}

	pieceSize := DefaultChunkSize
	fragSize := DefaultFragSize

	/* Invalid ranges */
	rangeM := backend.GetRangeMatrix(0, -1, pieceSize, fragSize)
	assert.Nil(t, rangeM)

	rangeM = backend.GetRangeMatrix(0, 10*fragSize, pieceSize, fragSize)
	assert.Nil(t, rangeM)

	/* Range is aligned on groups. */
	startIncl := 0
	endIncl := 0
	rangeM = backend.GetRangeMatrix(startIncl, endIncl, pieceSize, fragSize)

	assert.Equal(t, rangeM.FragCount, 1)
	assert.Equal(t, rangeM.FragFirstIncl, 0)
	assert.Equal(t, rangeM.ReqStartIncl, startIncl)
	assert.Equal(t, rangeM.ReqEndIncl, endIncl)
	assert.Equal(t, rangeM.ReqEndIncl, endIncl)

	expectedTotalRead := (backend.headerSize + pieceSize)
	totalRead := rangeM.FragCount * (rangeM.InFragRangeEndExcl - rangeM.InFragRangeStartIncl)
	assert.Equal(t, expectedTotalRead, totalRead)

	/* Range spanning a single group */
	startIncl = 0
	endIncl = backend.K*pieceSize - 1
	rangeM = backend.GetRangeMatrix(startIncl, endIncl, pieceSize, fragSize)
	assert.Equal(t, rangeM.FragCount, backend.K)
	assert.Equal(t, rangeM.FragFirstIncl, 0)
	assert.Equal(t, rangeM.ReqStartIncl, startIncl)
	assert.Equal(t, rangeM.ReqEndIncl, endIncl)

	expectedTotalRead = backend.K * (backend.headerSize + pieceSize)
	totalRead = rangeM.FragCount * (rangeM.InFragRangeEndExcl - rangeM.InFragRangeStartIncl)
	assert.Equal(t, expectedTotalRead, totalRead)

	/* Range spanning multiple groups, this fallback to read two full groups. */
	startIncl = (backend.K - 1) * pieceSize
	endIncl = backend.K * pieceSize
	rangeM = backend.GetRangeMatrix(startIncl, endIncl, pieceSize, fragSize)
	assert.Equal(t, rangeM.FragFirstIncl, 0)
	assert.Equal(t, rangeM.FragCount, 4)
	assert.Equal(t, rangeM.ReqStartIncl, startIncl)
	assert.Equal(t, rangeM.ReqEndIncl, endIncl)

	expectedTotalRead = 2 * backend.K * (backend.headerSize + pieceSize)
	totalRead = rangeM.FragCount * (rangeM.InFragRangeEndExcl - rangeM.InFragRangeStartIncl)
	assert.Equal(t, expectedTotalRead, totalRead)
}

func TestGetRangeMatrix(t *testing.T) {
	type testCase struct {
		name              string
		start             int
		end               int
		chunksize         int
		fragSize          int
		payloadSize       int
		expectedFragStart int
		expectedFragEnd   int
		expectedDecStart  int
		expectedDecEnd    int
	}

	backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: 2, M: 1})
	defer func() {
		_ = backend.Close()
	}()

	testCases := []testCase{
		{
			name:              "First 128 bytes, 100k payload",
			start:             0,
			end:               128,
			chunksize:         32768,
			fragSize:          1048576,
			payloadSize:       100000,
			expectedFragStart: 0,
			expectedFragEnd:   32768 + backend.headerSize,
			expectedDecStart:  0,
			expectedDecEnd:    128,
		},
		{
			name:              "First 128 bytes, 1MB payload",
			start:             0,
			end:               128,
			chunksize:         32768,
			fragSize:          1048576,
			payloadSize:       1000000,
			expectedFragStart: 0,
			expectedFragEnd:   32768 + backend.headerSize,
			expectedDecStart:  0,
			expectedDecEnd:    128,
		},
		{
			name:              "64k Block in the middle, 100k payload",
			start:             32768,
			end:               32768 + 65536,
			chunksize:         32768,
			fragSize:          1048576,
			payloadSize:       100000,
			expectedFragStart: 0,
			expectedFragEnd:   65536 + 2*backend.headerSize,
			expectedDecStart:  32768,
			expectedDecEnd:    65536 + 32768,
		},
		{
			name:              "64k Block in the middle, 1MB payload",
			start:             500000,
			end:               500000 + 65536,
			chunksize:         32768,
			fragSize:          1048576,
			payloadSize:       1000000,
			expectedFragStart: 7 * (32768 + 80),
			expectedFragEnd:   7*(32768+80) + 65536 + backend.headerSize*2,
			expectedDecStart:  500000 - 458752,
			expectedDecEnd:    500000 - 458752 + 65536,
		},
		{
			name:              "Last 80 bytes, 100k payload",
			start:             100000 - 80,
			end:               100000,
			chunksize:         32768,
			fragSize:          1048576,
			payloadSize:       100000,
			expectedFragStart: 32768 + 80,
			expectedFragEnd:   32768 + 80 + 80 + 32768,
			expectedDecStart:  100000 - 65536 - 80,
			expectedDecEnd:    100000 - 65536,
		},
		{
			name:              "Last 80 bytes, 1MB payload",
			start:             1000000 - 80,
			end:               1000000,
			chunksize:         32768,
			fragSize:          1048576,
			payloadSize:       1000000,
			expectedFragStart: (1000000 / 32768 / 2) * (32768 + 80),
			expectedFragEnd:   (1000000/32768/2)*(32768+80) + 32768 + backend.headerSize,
			expectedDecStart:  1000000 - ((1000000 / 32768) * 32768) - 80,
			expectedDecEnd:    1000000 - ((1000000 / 32768) * 32768),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			length := tc.end - tc.start
			rm := backend.GetRangeMatrix(tc.start, tc.end, tc.chunksize, tc.fragSize)
			require.NotNil(t, rm, "GetRangeMatrix returned nil")
			assert.Equal(t, tc.expectedFragStart, rm.InFragRangeStartIncl, "FragRangeStart mismatch")
			assert.Equal(t, tc.expectedFragEnd, rm.InFragRangeEndExcl, "FragRangeEnd mismatch")
			assert.Equal(t, tc.expectedDecStart, rm.DecodedRangeStartIncl, "DecodedRangeStart mismatch")
			assert.Equal(t, tc.expectedDecEnd, rm.DecodedRangeStartIncl+length, "DecodedRangeEnd mismatch")
		})
	}
}

func TestRange(t *testing.T) {
	testCases := []struct {
		name         string
		useNewFormat bool
	}{
		{"OldFormat", false},
		{"NewFormat", true},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			testRangeHelper(t, testCase.useNewFormat)
		})
	}
}

func testRangeHelper(t *testing.T, useNewFormat bool) {
	backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: 2, M: 1})
	defer func() {
		_ = backend.Close()
	}()

	chunkSize := 32768
	size := 1020300

	rm := backend.GetRangeMatrix(10, 20, chunkSize, size)
	require.NotNil(t, rm, "GetRangeMatrix returned nil")

	buf := make([]byte, size)
	for i := range buf {
		buf[i] = byte('A' + i%26)
	}

	bm := NewBufferMatrix(chunkSize, len(buf), backend.K)
	if !useNewFormat {
		bm.UseOldFormat()
	}
	_, err := io.Copy(bm, bytes.NewReader(buf))
	require.NoError(t, err)
	bm.Finish()
	encodedData, err := backend.EncodeMatrixWithBufferMatrix(bm, chunkSize)
	require.NoError(t, err)
	defer encodedData.Free()
	stripes := make([][]byte, backend.K+backend.M)
	for i := range backend.K + backend.M {
		stripes[i] = encodedData.Data[i]
		stripes[i] = stripes[i][rm.InFragRangeStartIncl:rm.InFragRangeEndExcl]
	}

	// Test decode
	decodedData, err := backend.DecodeMatrix(stripes, chunkSize)
	require.NoError(t, err)
	defer decodedData.Free()
	assert.Equal(t, buf[10:20], decodedData.Data[rm.DecodedRangeStartIncl:rm.DecodedRangeStartIncl+20-10], "Decoded data mismatch")

	// Test repair
	decodedData2, err := backend.DecodeMatrix(stripes[1:], chunkSize)
	require.NoError(t, err)
	defer decodedData2.Free()
	assert.Equal(t, buf[10:20], decodedData2.Data[rm.DecodedRangeStartIncl:rm.DecodedRangeStartIncl+20-10], "Decoded data mismatch")

	// Test last 80 bytes
	rm = backend.GetRangeMatrix(size-80, size, chunkSize, size)
	require.NotNil(t, rm, "GetRangeMatrix returned nil")

	for i := range backend.K + backend.M {
		stripes[i] = encodedData.Data[i]
		stripes[i] = stripes[i][rm.InFragRangeStartIncl:rm.InFragRangeEndExcl]
	}

	// Test decode
	decodedData3, err := backend.DecodeMatrix(stripes, chunkSize)
	require.NoError(t, err)
	defer decodedData3.Free()
	assert.Equal(t, buf[size-80:size], decodedData3.Data[rm.DecodedRangeStartIncl:rm.DecodedRangeStartIncl+80], "Decoded data mismatch")
	// Test repair
	decodedData4, err := backend.DecodeMatrix(stripes[1:], chunkSize)
	require.NoError(t, err)
	defer decodedData4.Free()
	assert.Equal(t, buf[size-80:size], decodedData4.Data[rm.DecodedRangeStartIncl:rm.DecodedRangeStartIncl+80], "Decoded data mismatch")
}

// TestFormatOldNew tests the compatibility of the new format with the old one
// It uses the buffer matrix to encode the data in the old/new format and
// then decodes it using the backend. It checks that the data is the same
// and that the format is correct.
func TestFormatOldNew(t *testing.T) {
	testCases := []struct {
		useNewFormat bool
		k, n         int
	}{
		{true, 2, 1},
		{true, 5, 1},
		{false, 2, 1},
		{false, 5, 1},
	}
	for _, testCase := range testCases {
		t.Run(fmt.Sprintf("%v-%d-%d", testCase.useNewFormat, testCase.k, testCase.n), func(t *testing.T) {
			// use buffermatrix to storage format in new format and see if we can decode it
			backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: testCase.k, M: testCase.n})
			require.NoError(t, err)
			defer backend.Close()
			buf := bytes.Repeat([]byte("A"), 1024*1024+rand.Intn(1024*1024)) //nolint:gosec

			bm := NewBufferMatrix(32768, len(buf), backend.K)
			if testCase.useNewFormat {
				bm.UseNewFormat()
			}
			_, err = io.Copy(bm, bytes.NewReader(buf))
			require.NoError(t, err)
			bm.Finish()

			require.Equal(t, len(buf), bm.Length())

			e, err := backend.EncodeMatrixWithBufferMatrix(bm, 32768)
			require.NoError(t, err)
			defer e.Free()

			// check the format / first 80 is the header, lets check it
			for i := range len(e.Data) {
				hdr := e.Data[i][0:80]

				var f fragheader
				err = f.UnmarshalBinary(hdr)
				require.NoError(t, err)
				require.Equal(t, 32768, int(f.meta.size))
				require.Equal(t, 32768*testCase.k, int(f.meta.origDataSize)) //nolint:gosec
			}
			// case 1; fast decode
			ddata, err := backend.DecodeMatrix(e.Data, 32768)
			require.NoError(t, err)
			require.Equal(t, buf, ddata.Data)
			defer ddata.Free()
			// case 2: missing data
			rdata, err := backend.ReconstructMatrix(e.Data[1:], 0, 32768)
			require.NoError(t, err)
			require.Equal(t, e.Data[0], rdata.Data)
			defer rdata.Free()
			// case 3: slow decode
			ddata2, err := backend.DecodeMatrix(e.Data[1:], 32768)
			require.NoError(t, err)
			require.Equal(t, buf, ddata2.Data)
			defer ddata2.Free()
			// case 4: rebuild missing coding
			require.Equal(t, testCase.k, len(e.Data[:testCase.k]))
			rdata2, err := backend.ReconstructMatrix(e.Data[:testCase.k], testCase.k, 32768)
			require.NoError(t, err)
			require.Equal(t, e.Data[testCase.k], rdata2.Data)
		})
	}
}

// duplicate fragment_header_t from libec
type fragheader struct {
	meta           fragmeta
	magic          uint32
	libecVersion   uint32
	metadataChksum uint32
	padding        [9]byte
}

func (f *fragheader) UnmarshalBinary(data []byte) error {
	if len(data) != 80 {
		return fmt.Errorf("invalid size for fragment header: %d", len(data))
	}
	if err := f.meta.UnmarshalBinary(data[0:63]); err != nil {
		return err
	}
	f.magic = binary.BigEndian.Uint32(data[63:67])
	f.libecVersion = binary.BigEndian.Uint32(data[67:71])
	f.metadataChksum = binary.BigEndian.Uint32(data[71:75])
	copy(f.padding[:], data[75:80])
	return nil
}

func (f *fragmeta) UnmarshalBinary(data []byte) error {
	if len(data) != 63 {
		return fmt.Errorf("invalid size for fragment metadata: %d", len(data))
	}
	f.idx = binary.BigEndian.Uint32(data[0:4])
	f.size = binary.LittleEndian.Uint32(data[4:8])
	f.fragBackendMetadataSize = binary.LittleEndian.Uint32(data[8:12])
	f.origDataSize = binary.LittleEndian.Uint64(data[12:20])
	f.checksumType = data[20]
	copy(f.checksum[:], data[21:53])
	f.checksumMismatch = data[53]
	f.backendID = data[54]
	f.backendVersion = binary.BigEndian.Uint32(data[55:59])
	return nil
}

type fragmeta struct {
	idx                     uint32
	size                    uint32
	fragBackendMetadataSize uint32
	origDataSize            uint64
	checksumType            uint8
	checksum                [32]byte
	checksumMismatch        uint8
	backendID               uint8
	backendVersion          uint32
}
