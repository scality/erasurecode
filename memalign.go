package erasurecode

import (
	"errors"
	"unsafe"
)

func getAlignDifference(b []byte, align int) int {
	return int(uintptr(unsafe.Pointer(&b[0])) & uintptr(align-1))
}

const defaultAlign = 16

func memalign(n int, align int) ([]byte, error) {
	if n == 0 {
		return nil, errors.New("invalid size")
	}

	buf := make([]byte, n+align)

	var offSet int
	// check alignment
	if offSet = getAlignDifference(buf, align); offSet == 0 {
		// Success
		return buf[:n], nil
	}
	return buf[offSet : offSet+n], nil
}
