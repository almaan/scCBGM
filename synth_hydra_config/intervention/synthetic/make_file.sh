#!/bin/bash
for ii in {0..4}; do cp _template.yaml intervention_$ii.yaml; sed -i "s/\$NUM/$ii/g" intervention_$ii.yaml; done
