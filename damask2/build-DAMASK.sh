#!/bin/bash
set -e

#INTELROOT=$( which ifort | sed -e 's:bin/\w*/*ifort::' )
#source $INTELROOT/bin/compilervars.sh  intel64

function getsrc () {
  local filename=$(basename "$1")
  local dirname=${filename%%-*}
  [ -d ${dirname}*/ ] || ( wget "$1" && tar -xf "$filename" && rm "$filename" )
}

getsrc http://www.fftw.org/fftw-3.3.8.tar.gz
getsrc http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.9.4.tar.gz
getsrc https://damask.mpie.de/pub/Download/Current/DAMASK-2.0.2.tar.xz

FFTWROOT=$(   cd fftw*/   && pwd)
PETSCROOT=$(  cd petsc*/  && pwd)
DAMASKROOT=$( cd DAMASK*/ && pwd)

FFTWPRE="${FFTWROOT}/deploy_avx"

export PETSC_ARCH=linux-gnu-intel

ls "$MKLROOT" "$FFTWROOT" "$PETSCROOT" "$DAMASKROOT" > /dev/null

function build_petsc () {(
    cd "$PETSCROOT"
    unset PETSC_DIR
    mkdir -p "$PETSC_ARCH"
    ./configure PETSC_ARCH=$PETSC_ARCH \
         --with-blaslapack-dir="$MKLROOT" \
         --with-cc=mpicc --with-fc=mpif90 --with-cxx=mpicxx \
         --with-fftw=1   --with-fftw-dir="$FFTWPRE" \
         --with-x=0
    make PETSC_DIR=$PWD PETSC_ARCH=$PETSC_ARCH all test

    export PETSC_DIR=$PWD
    export PETSC_ARCH=$PETSC_ARCH
    [ -d $PETSC_DIR/$PETSC_ARCH ] || exit 1
)}

function build_damask () {(
    [ -z "$PETSC_DIR" ] && export PETSC_DIR=$PETSCROOT
    cd $DAMASKROOT
    mkdir -p build && cd ./build
    cmake --version
    cmake -DDAMASK_SOLVER=SPECTRAL -DDAMASK_ROOT=$DAMASKROOT ../
    make --no-print-directory -ws all
    make install
)}

#http://www.fftw.org/fftw3_doc/Installation-on-Unix.html#Installation-on-Unix
function build_fftw () {(
    cd "$FFTWROOT"
    ./configure --enable-mpi    \
                --enable-openmp \
                --enable-avx    \
                --enable-shared \
                --prefix=$PWD/deploy_avx
    mkdir -p $PWD/deploy_avx
    make
    make install
)}

[ -f "$FFTWPRE"/lib*/libfftw3.so ]              || build_fftw
[ -f "$PETSCROOT/$PETSC_ARCH"/lib/libpetsc.so ] || build_petsc
[ -f $DAMASKROOT/bin/DAMASK_spectral ]          || build_damask

# test
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PETSCROOT/$PETSC_ARCH/lib" $DAMASKROOT/bin/DAMASK_spectral --help
