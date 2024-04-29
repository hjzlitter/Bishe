    from ovito.io import *
    from ovito.data import *
    from ovito.modifiers import *
    from ovito.pipeline import *
    import numpy as np

    file_list = ["dump.noaddFe", "dump_temp.600.1"]
    pipeline = import_file(file_list[1])
    # reference_modifier = ReferenceConfigurationModifier(reference_data=reference_pipeline.source)
    ws = WignerSeitzAnalysisModifier(per_type_occupancies = True, affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference)
    ws.reference.load(file_list[0])
    selection = ExpressionSelectionModifier(expression = 'Occupancy ==1 ')
    delection = DeleteSelectedModifier()

    # pipeline.modifiers.append(reference_modifier)
    pipeline.modifiers.append(ws)
    pipeline.modifiers.append(selection)
    pipeline.modifiers.append(delection)
    #for frame_index in range(2):
    #	data = pipeline.compute(frame_index)
    export_file(pipeline, "outputfile.dump", "lammps/dump", multiple_frames = True, columns = ["Particle Identifier","Position.X", "Position.Y", "Position.Z"])

#print(data.attributes)
#print(data.particles.count)
#print(pipeline.source.num_frames)
#data = pipeline.compute(3)
