#!/bin/bash
for ii in {0..4}; do cp _template.yaml concept_$ii.yaml; sed -i "s/\$CONCEPT/$ii/g" concept_$ii.yaml; done
