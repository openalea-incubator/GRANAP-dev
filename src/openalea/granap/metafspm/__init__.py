"""
GRANAP's sections in metafspm's MPG. The organ classes (openalea.granap.root_class.RootAnatomy, ...) generate their
cross-section into an MPG; these modules hold what is MPG-specific:

    build       the section written into the MPG (Layers, Cells, protoplasts; the exporter's network)
    network     the section's network built in the MPG from its stored rings (walls, junctions, Connections)
    columns     the network's quantities as numeric columns (DataStructure variables)
    topology    the section's rings stored in the MPG, and its topology rebuilt from them
    cells       the cells as views of the MPG during the stages
    views       GRANAP's objects (cells, GeoDataFrame, networkx graph) rebuilt from the MPG
    component   GranapAnatomy, the metafspm structural component (anatomies of a population's segments)
    library     the anatomy templates by class, instantiated per segment
    vessels     vessel ids and the junctions between successive sections
"""
