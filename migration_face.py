
import numpy as np
# import ovito
from ovito import pipeline, io, modifiers
# import get_data

LATTICE_CONSTANT = 2.863
MIG_INDEX_PATH = "./zzx/data/MigIndex_{temperature}.npy"
MIG_FACE_PATH = "./zzx/data/MigFace_{temperature}.npy"
MIG_FACE_LABEL_PATH = "./zzx/data/MigFaceLabel_{temperature}_{time_len}.npy"
DUMP_ADDHE_PATH = "./MD/SingleHe/temper.{temperature}/dump.addHe.{temperature}"
DUMP_TEMP_PATH = "./MD/SingleHe/temper.{temperature}/dump_temp.{temperature}.{index}"


def GetMigrationIndex(
        p_temperature: int,
) -> np.ndarray:
    return np.load(MIG_INDEX_PATH.format(temperature=p_temperature))


def GetCellSize(
    p_temperature: int,  # 温度
):
    cell_size = np.zeros(3)
    with open(DUMP_ADDHE_PATH.format(temperature=p_temperature),
              'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin.readlines()[5:8]):
            cell_boundary = list(map(float, line.split()))
            cell_size[i] = cell_boundary[1] - cell_boundary[0]
    return cell_size


def NearestModify(
    p_pipeline: pipeline.Pipeline,
    p_num_neighbors: int = 4
):
    p_pipeline.modifiers.append(modifiers.ExpressionSelectionModifier(
        expression="ParticleType == 2"
    ))
    p_pipeline.modifiers.append(modifiers.ExpandSelectionModifier(
        mode=modifiers.ExpandSelectionModifier.ExpansionMode.Nearest,
        num_neighbors=p_num_neighbors
    ))
    p_pipeline.modifiers.append(modifiers.InvertSelectionModifier())
    p_pipeline.modifiers.append(modifiers.DeleteSelectedModifier())
    return


def GetNearestAtoms(
    p_temperature: int,
    p_index: int,
    p_cell_size: np.ndarray
):
    pipeline = io.import_file(DUMP_TEMP_PATH.format(
        temperature=p_temperature,
        index=p_index // 1000 + 1
    ))
    NearestModify(pipeline)
    data = pipeline.compute(p_index % 1000 + 1)
    type_of_atoms = data.particles.particle_types
    pos_of_Fe = np.array(data.particles.positions[type_of_atoms == 1])
    pos_of_He = np.array(data.particles.positions[type_of_atoms == 2]).squeeze(0)

    pos_of_Fe = np.select(
        [pos_of_Fe > pos_of_He + p_cell_size/2, pos_of_Fe < pos_of_He - p_cell_size/2],
        [pos_of_Fe - p_cell_size, pos_of_Fe + p_cell_size],
        default=pos_of_Fe
    )

    id_of_Fe = np.array(data.particles["Particle Identifier"][type_of_atoms == 1])
    return dict(zip(id_of_Fe, pos_of_Fe))


def GetMigrationFace(
    p_temperature: int,
    p_index: int,
    p_cell_size: np.ndarray
):
    tetrahedron_before = GetNearestAtoms(p_temperature, p_index-1, p_cell_size)
    tetrahedron_after = GetNearestAtoms(p_temperature, p_index, p_cell_size)

    id_of_face = np.intersect1d(
        list(tetrahedron_before.keys()),
        list(tetrahedron_after.keys())
    )
    migration_face = {
        id: tetrahedron_after[id] for id in id_of_face
    }
    return migration_face


def _GetAllMigrationFace(
    p_temperature: int
):
    # cell_size = GetCellSize(p_temperature)
    mig_index = GetMigrationIndex(p_temperature)
    migration_face = []
    for i in range(len(mig_index)):
        # np.save(
        #     "./zzx/data/temp/MigFace_{}_{}.npy".format(p_temperature, i),
        #     np.array(GetMigrationFace(p_temperature, mig_index[i], cell_size))
        # )

        migration_face.append(np.load(
            "./zzx/data/temp/MigFace_{}_{}.npy".format(p_temperature, i),
            allow_pickle=True
        ).item())
        # break
    migration_face = np.array(migration_face)

    np.save(
        MIG_FACE_PATH.format(temperature=p_temperature),
        migration_face
    )
    # return migration_face
    return


def _IsTheSecondNearestNeighbor(
    p_distance: float
):
    return p_distance > LATTICE_CONSTANT * 0.933012


def JudgeMigrationFaceLabel(
    p_mig_face_1: dict,
    p_mig_face_2: dict
) -> int:
    share_side = np.intersect1d(
        list(p_mig_face_1.keys()),
        list(p_mig_face_2.keys())
    )

    if len(share_side) < 2 or len(share_side) > 3:
        # 出错
        print(len(share_side))
        return -1
    elif len(share_side) == 3:
        # 情况0 从哪来，回哪去
        return 0
    elif _IsTheSecondNearestNeighbor(np.linalg.norm(
        p_mig_face_1[share_side[0]] - p_mig_face_1[share_side[1]]
    )):
        # 情况1
        return 1

    # 情况2、3
    uncoincident_points = np.setdiff1d(  # 1面中与2面不重合点
        list(p_mig_face_1.keys()),
        list(p_mig_face_2.keys())
    ).item()
    vertex = np.setdiff1d(  # 四面体的顶点
        list(p_mig_face_2.keys()),
        list(p_mig_face_1.keys())
    ).item()
    if _IsTheSecondNearestNeighbor(np.linalg.norm(
        p_mig_face_1[share_side[0]] - p_mig_face_1[uncoincident_points]
    )):
        base_point = share_side[1]  # 面1等腰三角形的顶点
        coincident_points = share_side[0]  # 1面中与2面另一个重合点
    else:
        base_point = share_side[0]  # 面1等腰三角形的顶点
        coincident_points = share_side[1]  # 1面中与2面另一个重合点

    if np.dot(
        np.cross(
            p_mig_face_1[coincident_points] - p_mig_face_1[base_point],
            p_mig_face_1[uncoincident_points] - p_mig_face_1[base_point]
        ),
        p_mig_face_2[vertex] - p_mig_face_1[base_point]
    ) > 0:
        return 2
    else:
        return 3


def _GetDataset(
    p_temperature: int,
    p_time_length: int
):
    mig_index = GetMigrationIndex(p_temperature)
    mig_face = np.load(MIG_FACE_PATH.format(temperature=p_temperature), allow_pickle=True)

    data_temp = []
    for i in range(1, len(mig_index)-2):
        if mig_index[i+1] - mig_index[i] >= p_time_length \
                and mig_index[i] - mig_index[i-1] >= p_time_length//4 \
                and mig_index[i+2] - mig_index[i+1] >= p_time_length//4:
            data_temp.append([
                mig_index[i], mig_index[i+1],
                JudgeMigrationFaceLabel(mig_face[i], mig_face[i+1])
            ])
    data_temp = np.array(data_temp)
    np.save(
        MIG_FACE_LABEL_PATH.format(temperature=p_temperature, time_len=p_time_length),
        data_temp
    )
    print(data_temp.shape[0])
    return data_temp


if __name__ == "__main__":
    TEMPERATURE = 400
    # _GetAllMigrationFace(TEMPERATURE)
    _GetDataset(TEMPERATURE, 16)
    pass
