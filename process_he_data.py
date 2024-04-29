import numpy as np
from ovito import io, modifiers


rc = 5.41           # The cutoff distance of Fe-He potential energy
eta_g1 = [1, 0.05]  # list of η/pi/Rc**2 for g1
eta_g2 = [1, 0.005]  # list of η/pi/Rc**2 for g2
lambda_g2 = [1, -1]  # list of λ for g2
kxi_g2 = [1, 2]  # list of ξ for g2
DUMP_ADDHE_PATH = "./MD/SingleHe/temper.{temperature}/dump.addHe.{temperature}"
TIMEDT_DATA_PATH = "./MD/SingleHe/temper.{temperature}/timedt.data.{temperature}"
DUMP_TEMP_PATH = "./MD/SingleHe/temper.{temperature}/dump_temp.{temperature}.{index}"


def GetCellSize(
    p_temperature: int,  # 温度
):
    cell_size = np.zeros(3)
    with open(DUMP_ADDHE_PATH.format(temperature=p_temperature), 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin.readlines()[5:8]):
            cell_boundary = list(map(float, line.split()))
            cell_size[i] = cell_boundary[1] - cell_boundary[0]
    return cell_size


def GetTimedtData(
    p_temperature: int
):
    t = []      # 时间序列(单位:ps)
    msd = []    # 单He的msd(均方位移)
    csp = []    # CSP(中心对称参数)
    xyz = []    # 单He的xyz坐标
    r_ = []     # 单He离原点距离
    v_xyz = []  # 单He的沿xyz坐标的速度分量
    v_ = []     # 单He的速度大小
    with open(TIMEDT_DATA_PATH.format(temperature=p_temperature), 'r', encoding='utf-8') as fin:
        for line in fin.readlines()[1:]:
            data = list(map(float, line.split()))
            t.append(data[0:1])
            msd.append(data[1:2])
            csp.append(data[2:8])
            xyz.append(data[8:11])
            r_.append(data[11:12])
            v_xyz.append(data[12:15])
            v_.append(data[15:16])
    t = np.array(t)
    msd = np.array(msd)
    csp = np.array(csp)
    xyz = np.array(xyz)
    r_ = np.sqrt(np.array(r_))
    v_xyz = np.array(v_xyz)
    v_ = np.sqrt(np.array(v_))
    angle = np.arccos(v_xyz / v_)
    return np.hstack((t, xyz, r_, v_xyz, v_, angle, msd, csp))


def _G1(
    p_x,
    p_eta,
    p_rc
):
    return 0.5 * np.exp(
        - np.pi * p_eta * (p_x / p_rc)**2
    ) * (np.cos(
        np.pi * p_x / p_rc
    ) + 1)


def _G2(
    p_x,
    p_y,
    p_eta,
    p_kxi,
    p_lambda,
    p_rc
):
    cos_theta_ijk = np.vdot(p_x, p_y) / (np.linalg.norm(p_x) * np.linalg.norm(p_y))
    # print(cos_theta_ijk)
    return 2**(1-p_kxi) * (
        1 + p_lambda * cos_theta_ijk
    )**p_kxi * _G1(np.linalg.norm(p_x), p_eta, p_rc) * _G1(np.linalg.norm(p_x), p_eta, p_rc) * _G1(np.linalg.norm(p_x - p_y), p_eta, p_rc)


def _G(
    p_pos
):
    temp_g = np.zeros((6))
    for i_pos in p_pos:
        temp_g[0] += _G1(
            np.linalg.norm(i_pos),
            eta_g1[0],
            rc
        )
        temp_g[1] += _G1(
            np.linalg.norm(i_pos),
            eta_g1[1],
            rc
        )
    for i in range(p_pos.shape[0]):
        for j in range(i+1, p_pos.shape[0]):
            temp_g[2] += _G2(
                p_pos[i],
                p_pos[j],
                eta_g2[0],
                kxi_g2[0],
                lambda_g2[0],
                rc
            )
            # break
            temp_g[3] += _G2(
                p_pos[i],
                p_pos[j],
                eta_g2[1],
                kxi_g2[1],
                lambda_g2[0],
                rc
            )
            temp_g[4] += _G2(
                p_pos[i],
                p_pos[j],
                eta_g2[0],
                kxi_g2[0],
                lambda_g2[1],
                rc
            )
            temp_g[5] += _G2(
                p_pos[i],
                p_pos[j],
                eta_g2[1],
                kxi_g2[1],
                lambda_g2[1],
                rc
            )
        # break
    return temp_g


def GetgDataOfPerTemp(
    p_file_path,
    p_cell_size
):
    # * 读取dump_temp文件
    pipeline = io.import_file(p_file_path)
    pipeline.modifiers.append(
        modifiers.ExpressionSelectionModifier(expression='ParticleType == 2')
    )
    pipeline.modifiers.append(
        modifiers.ExpandSelectionModifier(cutoff=rc)
    )
    pipeline.modifiers.append(modifiers.InvertSelectionModifier())
    pipeline.modifiers.append(modifiers.DeleteSelectedModifier())

    # * 计算g参数
    g_temp = []
    for i_frame in range(pipeline.source.num_frames):
        data_per_frame = pipeline.compute(i_frame)
        type_of_atoms = data_per_frame.particles.particle_types
        pos_of_Fe = np.array(data_per_frame.particles.positions[type_of_atoms == 1])
        pos_of_He = np.array(data_per_frame.particles.positions[type_of_atoms == 2]).squeeze(0)

        pos_of_Fe -= pos_of_He
        pos_of_Fe = np.select(
            [pos_of_Fe > p_cell_size/2, pos_of_Fe < -p_cell_size/2],
            [pos_of_Fe - p_cell_size, pos_of_Fe + p_cell_size],
            default=pos_of_Fe
        )
        g_temp.append(_G(pos_of_Fe))
        # break
    return g_temp


def GetgDataOfHeAtom(
    p_temperature: int
):
    g_of_He = []
    for i in range(1, 1 + 1):
        g_of_He += GetgDataOfPerTemp(
            DUMP_TEMP_PATH.format(temperature=p_temperature, index=i),
            GetCellSize(p_temperature)
        )[1:]
    g_of_He = np.stack(g_of_He, axis=0)
    # print(g_of_He.shape)
    return g_of_He


if __name__ == "__main__":
    TEMPERATURE = 400
    data_of_He = np.hstack((
        GetTimedtData(TEMPERATURE)[:1000],
        GetgDataOfHeAtom(TEMPERATURE)
    ))
    print(data_of_He.shape)
    np.save(
        "./zzx/data/HeData1_{}".format(TEMPERATURE),
        data_of_He
    )
    pass
