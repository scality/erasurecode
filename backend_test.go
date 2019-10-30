package erasurecode

import (
	"bytes"
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

var validParams = []Params{
	{Name: "liberasurecode_rs_vand", K: 2, M: 1},
	{Name: "liberasurecode_rs_vand", K: 10, M: 4},
	{Name: "liberasurecode_rs_vand", K: 4, M: 3},
	{Name: "liberasurecode_rs_vand", K: 8, M: 4},
	{Name: "liberasurecode_rs_vand", K: 15, M: 4},
	{Name: "isa_l_rs_vand", K: 2, M: 1},
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
		for patternIndex, pattern := range testPatterns {
			data, err := backend.Encode(pattern)
			if err != nil {
				t.Errorf("Error encoding %v: %q", params, err)
				break
			}

			expectedVersion := GetVersion()
			frags := data.Data
			for index, frag := range frags {
				info := GetFragmentInfo(frag)
				if info.Index != index {
					t.Errorf("Expected frag %v to have index %v; got %v", index, index, info.Index)
				}
				if info.Size != len(frag)-80 { // 80 == sizeof (struct fragment_header_s)
					t.Errorf("Expected frag %v to have size %v; got %v", index, len(frag)-80, info.Size)
				}
				if info.OrigDataSize != uint64(len(pattern)) {
					t.Errorf("Expected frag %v to have orig_data_size %v; got %v", index, len(pattern), info.OrigDataSize)
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

			decode := func(frags [][]byte, description string) bool {
				decoded, err := backend.Decode(frags)
				if err != nil {
					t.Errorf("%v: %v: %q for pattern %d", description, backend, err, patternIndex)
					return false
				} else if !bytes.Equal(decoded.Data, pattern) {
					t.Errorf("%v: Expected %v to roundtrip pattern %d, got %q", description, backend, patternIndex, decoded.Data)
					return false
				}
				decoded.Free()
				return true
			}

			var good bool
			good = decode(frags, "all frags")
			good = good && decode(shuf(frags), "all frags, shuffled")
			good = good && decode(frags[:params.K], "data frags")
			good = good && decode(shuf(frags[:params.K]), "shuffled data frags")
			good = good && decode(frags[params.M:], "with parity frags")
			good = good && decode(shuf(frags[params.M:]), "shuffled parity frags")

			if !good {
				break
			}
			data.Free()
		}

		if _, err := backend.Decode([][]byte{}); err == nil {
			t.Errorf("Expected error when decoding from empty fragment array")
		}

		err = backend.Close()
		if err != nil {
			t.Errorf("Error closing backend %v: %q", backend, err)
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
		for patternIndex, pattern := range testPatterns {
			data, err := backend.Encode(pattern)
			frags := data.Data
			if err != nil {
				t.Errorf("Error encoding %v: %q", params, err)
			}

			reconstruct := func(recon_frags [][]byte, frag_index int, description string) bool {
				data, err := backend.Reconstruct(recon_frags, frag_index)
				if err != nil {
					t.Errorf("%v: %v: %q for pattern %d", description, backend, err, patternIndex)
					return false
				} else if !bytes.Equal(data, frags[frag_index]) {
					t.Errorf("%v: Expected %v to roundtrip pattern %d, got %q", description, backend, patternIndex, data)
					return false
				}
				return true
			}

			var good bool
			good = reconstruct(shuf(frags[:params.K]), params.K+params.M-1, "last frag from data frags")
			good = good && reconstruct(shuf(frags[params.M:]), 0, "first frag with parity frags")
			if !good {
				break
			}
			data.Free()
		}

		if _, err := backend.Reconstruct([][]byte{}, 0); err == nil {
			t.Errorf("Expected error when reconstructing from empty fragment array")
		}

		err = backend.Close()
		if err != nil {
			t.Errorf("Error closing backend %v: %q", backend, err)
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
		for patternIndex, pattern := range testPatterns {
			data, err := backend.Encode(pattern)
			if err != nil {
				t.Errorf("Error encoding %v: %q", params, err)
				continue
			}
			frags := data.Data
			for index, frag := range frags {
				if backend.IsInvalidFragment(frag) {
					t.Errorf("%v: frag %v unexpectedly invalid for pattern %d", backend, index, patternIndex)
				}
				fragCopy := make([]byte, len(frag))
				copy(fragCopy, frag)

				// corrupt the frag
				corruptedByte := rand.Intn(len(frag))
				for 71 <= corruptedByte && corruptedByte < 80 {
					// in the alignment padding -- try again
					corruptedByte = rand.Intn(len(frag))
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
			data.Free()
		}
		err = backend.Close()
		if err != nil {
			t.Errorf("Error closing backend %v: %q", backend, err)
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
			Name: "liberasurecode_rs_vand",
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
	}{
		struct {
			name     string
			testFunc func()
		}{
			"Reconstruct",
			func() {
				encoded, err := backend.Encode(input)

				if err != nil {
					t.Fatal("cannot encode data")
					return
				}
				vect := encoded.Data
				defer encoded.Free()

				oldData := vect[0][:] // force a copy

				data, err := backend.Reconstruct(vect[1:3], 0)
				if err != nil {
					t.Fatalf("cannot reconstruct data, %s", err)
					return
				}

				if len(data) != len(oldData) {
					t.Fatal("reconstructing failed")
					return
				}
			},
		},
		struct {
			name     string
			testFunc func()
		}{
			"Decode",
			func() {
				encoded, err := backend.Encode(input)

				if err != nil {
					t.Fatal("cannot encode data")
					return
				}
				defer encoded.Free()
				vect := encoded.Data

				decoded, err := backend.Decode(vect[0:2])
				if err != nil {
					t.Fatalf("cannot decode data: %v", err)
					return
				}
				defer decoded.Free()
				data := decoded.Data
				if len(data) != len(input) {
					t.Fatal("decoding failed")
					return
				}
			},
		},
	}

	nbRoutines := 500

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
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
		encoded, err := backend.Encode(buf)

		if err != nil {
			b.Fatal(err)
		}
		encoded.Free()
	}
}

const DefaultChunkSize = 32768

func BenchmarkDecodeM(b *testing.B) {
	backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})

	buf := bytes.Repeat([]byte("A"), 1024*1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encoded, err := backend.EncodeMatrix(buf, DefaultChunkSize)

		if err != nil {
			b.Fatal(err)
		}
		defer encoded.Free()

		decoded, err := backend.DecodeMatrix(encoded.Data, DefaultChunkSize)
		if err != nil {
			b.Fatal(err)
		}
		if decoded != nil {
			if decoded.Free != nil {
				decoded.Free()
			}
		} else {
			b.Fatal("decoded is nil")
		}
	}
}

func BenchmarkDecodeMSlow(b *testing.B) {
	backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})
	buf := bytes.Repeat([]byte("A"), 1024*1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encoded, err := backend.EncodeMatrix(buf, DefaultChunkSize)

		if err != nil {
			b.Fatal(err)
		}
		defer encoded.Free()

		decoded, err := backend.decodeMatrixSlow(encoded.Data, DefaultChunkSize)
		if err != nil {
			b.Fatal(err)
		}

		defer decoded.Free()
	}
}

func BenchmarkEncodeM(b *testing.B) {
	backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})

	buf := bytes.Repeat([]byte("A"), 1024*1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encoded, err := backend.EncodeMatrix(buf, 32768)

		if err != nil {
			b.Fatal(err)
		}
		encoded.Free()
	}
}

func BenchmarkDecode(b *testing.B) {
	backend, _ := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})

	buf := bytes.Repeat([]byte("A"), 1024*1024)
	res, _ := backend.Encode(buf)

	defer res.Free()

	for i := 0; i < b.N; i++ {
		decoded, err := backend.Decode(res.Data)

		if err != nil {
			b.Fatal(err)
		}
		decoded.Free()
	}
}

func TestEncodeM(t *testing.T) {
	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: 4, M: 2, W: 8, HD: 5})

	if err != nil {
		t.Fatalf("cannot init backend: (%v)", err)
	}

	buf := make([]byte, 1024*1024)
	for i := 0; i < len(buf); i++ {
		buf[i] = byte(65 + i%26)
	}

	// All our sub tests case. Each {X,Y} represents respecitvely the chunking unit (size of each subpart)
	// and the data we want to read when we decode
	testParams := [][]int{{4096, 4097}, {4097, 4096}, {4096, len(buf)}, {4096, 4096}}

	for _, param := range testParams {
		p := param
		testName := fmt.Sprintf("TestEncodeB-%d-%d", p[0], p[1])
		t.Run(testName, func(t *testing.T) {
			// Do the matrix encoding
			result, err := backend.EncodeMatrix(buf, p[0])

			defer result.Free()

			if err != nil {
				t.Errorf("failed to encode %+v", err)
			}

			// Do the matrix decoding. It should work fastly because we have
			// all data fragments. After, we check that our linearized buffer
			// contains expected data
			ddata, err := backend.DecodeMatrix(result.Data, p[0])
			if ok := checkData(ddata.Data); ok == false {
				t.Errorf("bad matrix decoding")
			}

			defer ddata.Free()

			/* now do the same but with the slow path*/
			/* we will run a matrix decoding but withtout some data part, to enforce repairing*/
			var vect [][]byte
			vect = append(vect, result.Data[2])
			vect = append(vect, result.Data[3])
			vect = append(vect, result.Data[4])
			vect = append(vect, result.Data[5])

			ddata2, err := backend.decodeMatrixSlow(vect, p[0])
			if ok := checkData(ddata2.Data); ok == false {
				t.Errorf("bad matrix repairing")
			}

			/*
			 * now we will do the same but we should failed because the chunksize provided
			 * to decode the data is not the same used by encode function
			 */
			_, err = backend.decodeMatrixSlow(result.Data, p[0]+1)
			if err == nil {
				t.Errorf("no error during decoding whereas bad params were provided")
			}
		})
	}
}

// checkData reads a buffer and check that we have a round robin sequence of [A-Z] characters
func checkData(data []byte) bool {
	for i := 0; i < len(data)-1; i++ {
		if data[i] != 'Z' {
			if data[i] != data[i+1]-1 {
				return false
			}
		} else if data[i+1] != 'A' {
			return false
		}
	}
	return true
}

func TestMatrixBounds(t *testing.T) {
	backend, err := InitBackend(Params{Name: "isa_l_rs_vand", K: 3, M: 1, W: 8, HD: 5})

	if err != nil {
		t.Fatalf("cannot init backend: (%v)", err)
	}

	// All our sub tests case. Each {X,Y} represents respecitvely the chunking unit (size of each subpart)
	// and the data we want to read when we decode
	testParams := [][]int{{0, 95, 32, 0, 112}, {10, 97, 32, 0, 224}, {97, 98, 32, 112, 224}, {0, 32*backend.K*2 + 1 + 1, 32, 0, 336}}

	for _, param := range testParams {
		p := param
		testName := fmt.Sprintf("TestEMatrixBounds-%d-%d", p[0], p[1])
		t.Run(testName, func(t *testing.T) {
			start, end := backend.GetRangeMatrix(p[0], p[1], p[2])
			if start != p[3] || end != p[4] {
				t.Errorf("start is %d instead of %d and end is %d instead of %d", start, p[3], end, p[4])
			}

		})
	}
}
