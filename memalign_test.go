package erasurecode

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMemalign(t *testing.T) {
	for i := 1; i < 16; i++ {
		align := 1 << i
		fmt.Println(align)
		ptr, err := memalign(1000, int(align))
		assert.NoError(t, err)
		assert.Zero(t, getAlignDifference(ptr, align))
	}
}
