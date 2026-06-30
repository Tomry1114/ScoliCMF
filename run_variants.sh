cd ~/ScoliCMF
srun -p debug --gres=gpu:1 --time=00:28:00 --pty bash -lc 'cd ~/ScoliCMF; P=~/.conda/envs/AgentOCR/bin/python
{ $P adoc_variants.py --geo 1 --photo 0 --center gauss --tag geo \
 && $P adoc_variants.py --geo 0 --photo 1 --center gauss --tag photo \
 && $P adoc_variants.py --geo 1 --photo 1 --center none --tag cen_none \
 && $P adoc_variants.py --geo 1 --photo 1 --center strong --tag cen_strong \
 && $P adoc_variants.py --geo 1 --photo 1 --center gauss --tag cen_gauss \
 && $P gate_geophoto.py ; } 2>&1 | tee ~/ScoliCMF/exp23gate.out'
