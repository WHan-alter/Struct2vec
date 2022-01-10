## About the code

The code offers a way in strcture embedding. It first translates the pdb into a graph (a.a.-based), then use a GAT for feature grabbing.

Example code:
python struct_embed.py --cuda --file_data ../data/covid_antibody/merged_antibody_set.jsonl --features full --name full

This code is rewritten and originally from the github repository:
https://github.com/jingraham/neurips19-graph-protein-design 
