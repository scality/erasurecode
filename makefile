DEPDIR=$(PWD)/deps
BUILDDIR=$(PWD)/build

LIBECSRC=$(DEPDIR)/liberasurecode
ISALSRC=$(DEPDIR)/isa-l
GFCOMPLETESRC=$(DEPDIR)/gf-complete
JERASURESRC=$(DEPDIR)/jerasure

.PHONY: default test clean pretty cmds

default: $(BUILDDIR)/lib/liberasurecode.a $(BUILDDIR)/lib/libisal.a $(BUILDDIR)/lib/libJerasure.la cmds
	PKG_CONFIG_PATH=$(BUILDDIR)/lib/pkgconfig \
	go build

test: $(BUILDDIR)/lib/liberasurecode.a $(BUILDDIR)/lib/libisal.a $(BUILDDIR)/lib/libJerasure.la
	DYLIB_LIBRARY_PATH=$(BUILDDIR)/lib \
	LD_LIBRARY_PATH=$(BUILDDIR)/lib \
	PKG_CONFIG_PATH=$(BUILDDIR)/lib/pkgconfig \
	go test -v

cmds: ec-split ec-info

ec-split: $(PWD)/cmd/ec-split/main.go $(PWD)/backend.go $(PWD)/streaming.go
	PKG_CONFIG_PATH=$(BUILDDIR)/lib/pkgconfig \
	go build github.com/scality/erasurecode/cmd/ec-split

ec-info: $(PWD)/cmd/ec-info/main.go $(PWD)/backend.go $(PWD)/streaming.go
	PKG_CONFIG_PATH=$(BUILDDIR)/lib/pkgconfig \
	go build github.com/scality/erasurecode/cmd/ec-info

clean:
	rm -rf $(BUILDDIR) $(DEPDIR)

pretty:
	find $(PWD) -name '*.go' | xargs gofmt -l -w

$(ISALSRC)/autogen.sh:
	git clone https://github.com/01org/isa-l.git $(ISALSRC)

$(LIBECSRC)/autogen.sh:
	git clone https://github.com/scality/liberasurecode.git $(LIBECSRC)

$(JERASURESRC)/configure.ac:
	git clone https://github.com/ceph/jerasure.git $(JERASURESRC)

$(GFCOMPLETESRC)/autogen.sh:
	git clone https://github.com/ceph/gf-complete.git $(GFCOMPLETESRC)

$(PWD)/deps/%/configure: $(PWD)/deps/%/autogen.sh
	cd $(@D) && ./autogen.sh

$(JERASURESRC)/configure: $(JERASURESRC)/configure.ac
	cd $(@D) && autoreconf --force --install

$(PWD)/deps/%/Makefile: $(PWD)/deps/%/configure
	cd $(@D) && ./configure --prefix=$(BUILDDIR) \
	LDFLAGS=-L$(BUILDDIR)/lib CFLAGS=-I$(BUILDDIR)/include


$(BUILDDIR)/lib/liberasurecode.a: $(LIBECSRC)/Makefile
	cd $(LIBECSRC) && make install

$(BUILDDIR)/lib/libisal.a: $(ISALSRC)/Makefile
	cd $(ISALSRC) && make install

$(BUILDDIR)/lib/libgf_complete.a: $(GFCOMPLETESRC)/Makefile
	cd $(GFCOMPLETESRC) && make install

$(BUILDDIR)/lib/libJerasure.la: $(BUILDDIR)/lib/libgf_complete.a $(JERASURESRC)/Makefile
	cd $(JERASURESRC) && make install
