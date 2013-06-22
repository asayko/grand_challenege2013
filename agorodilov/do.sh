#!/bin/bash
for i in {0..9}
do
   ./imgs2sifts_vl ${i} < part.${i}
done
