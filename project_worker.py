import numpy as np
import sys, os
import math
import shutil
import argparse

# --- 环境变量配置 ---
#pyexadis_paths = ['../../python', '../../lib', '../../core/pydis/python', '../../core/exadis/python/']
opendis_dir = '/home/dong1/OpenDiS'
pyexadis_paths = [
    f'{opendis_dir}/python', 
    f'{opendis_dir}/lib', 
    f'{opendis_dir}/core/pydis/python', 
    f'{opendis_dir}/core/exadis/python/'
]
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from framework.disnet_manager import DisNetManager
    from pyexadis_base import ExaDisNet, NodeConstraints, SimulateNetwork
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Remesh, Topology
    from pyexadis_utils import combine_networks, insert_infinite_line
except ImportError:
    pass

# --- 几何函数 ---
def create_infinite_line(cell, origin, burg_vec, slip_plane_normal, line_dir=None):
    nodes, segs = [], []
    nodes, segs = insert_infinite_line(cell, nodes, segs, burg=burg_vec, plane=slip_plane_normal, origin=origin, linedir=line_dir) 
    G = ExaDisNet(cell, nodes, segs)
    return DisNetManager(G)

def create_circular_loop(cell, center_pos, radius, nlinks, burg_vec, plane_normal, pin_all_nodes=False):
    theta = np.arange(nlinks) * 2.0 * np.pi / nlinks
    constraints = np.zeros_like(theta)
    if pin_all_nodes: constraints[:] = NodeConstraints.PINNED_NODE
    n_vec = np.array(plane_normal) / np.linalg.norm(plane_normal)
    helper = np.array([0.0, 0.0, 1.0]) if np.abs(n_vec[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, n_vec); u /= np.linalg.norm(u)
    v = np.cross(n_vec, u); v /= np.linalg.norm(v)
    t_col = theta[:, np.newaxis]
    xyz = np.array(center_pos)[np.newaxis, :] + radius * (np.cos(t_col) * u[np.newaxis, :] + np.sin(t_col) * v[np.newaxis, :])
    rn = np.column_stack((xyz, constraints))
    links = np.zeros((nlinks, 8))
    for i in range(nlinks):
        links[i,:] = np.array((i, (i+1)%nlinks, burg_vec[0], burg_vec[1], burg_vec[2], plane_normal[0], plane_normal[1], plane_normal[2]))
    G = ExaDisNet(cell, rn, links)
    return DisNetManager(G)

# --- 主逻辑 ---
def run_simulation(dist, radius, stress_value, medge, mscrew, run_id):
    Lbox = 1000.0
    cell = pyexadis.Cell(h=Lbox*np.eye(3), is_periodic=[True, True, True])
    
    center_line = np.array([Lbox * 0.5, Lbox * 0.5, Lbox * 0.5])
    center_loop = np.array([Lbox * 0.5 - dist, Lbox * 0.5, Lbox * 0.5])
    
    burg = 1.0/np.sqrt(2.0)*np.array([-1.,1.,0.]); b_line = burg/np.linalg.norm(burg)
    norma = np.array([1.,1.,1.]); n_line = norma/np.linalg.norm(norma)
    y = b_line; z = n_line; x = np.cross(y, z)/np.linalg.norm(np.cross(y, z))
    Rorient = np.array([x, y, z])
    b_line = np.matmul(Rorient, b_line); n_line = np.matmul(Rorient, n_line)
    
    N1 = create_infinite_line(cell, center_line, b_line, n_line, line_dir=b_line)
    b_prism = np.matmul(Rorient, np.array([1., 1., -1.0])/np.sqrt(3.0)) 
    n_prism = np.matmul(Rorient, np.array([1., 1., -1.0])/np.sqrt(3.0))
    
    # 【优化项】: 动态计算节点数，防止因半径变化导致网格极度变形
    nlinks = max(20, int(radius * 2 * np.pi / 15.0))
    N2 = create_circular_loop(cell, center_loop, radius, nlinks, b_prism, n_prism, pin_all_nodes=False)

    L_target = (2.0 * np.pi * radius) / nlinks
    dynamic_maxseg = max(2.0, L_target * 1.5)
    dynamic_minseg = max(1, L_target * 0.3)
    
    net = combine_networks([N1, N2])
    
    #state = {"burgmag": e-10, "mu": 160e9, "nu": 0.3, "a": 0.01, "maxseg": 20, "minseg": 5, "rann": 1,"crystal": 'fcc', "Rorient": Rorient}
    #state = {"burgmag": 2.5e-10, "mu": 76e9, "nu": 0.31, "a": 0.01, "maxseg": 20, "minseg": 5, "rann": 1,"crystal": 'fcc', "Rorient": Rorient}
    state = {
    "burgmag": 2.86e-10, 
    "mu": 27e9, 
    "nu": 0.34, 
    "a": 6, 
    "maxseg": dynamic_maxseg, 
    "minseg": dynamic_minseg,
    "rann": 3,
    "crystal": 'fcc', 
    "Rorient": Rorient,
    "nextdt": 1e-15,  # 初始试探步长 (设得比你原来的 1e-9 小一点，让它自己放大)
    "maxdt": 4e-14,    # 步长上限 (直接用你 Euler 能稳跑的 1e-9 作为安全阀值，防止穿模)
    "rtol": 0.1,      # 误差容限 (使用 LineTension 时，1.0 即可兼顾速度和准度) 
}

    level_1 = 'Al_400k_DDD'
    level_2 = f'run_R{radius}_Me{int(medge)}_Ms{int(mscrew)}'
    level_3 = f's{int(stress_value)}'
    output_dir = os.path.join(level_1, level_2, level_3)
    # 安全机制：由于同一个 level_2 下会有多次不同应力的二分法试探，
    # 我们只删除并重建当前的 level_3 (stress) 文件夹，以免清空之前跑完的应力数据。
    if os.path.exists(output_dir): 
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    calforce = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=32, cell=net.cell)
    mobility = MobilityLaw(mobility_law='FCC_0', state=state, Medge=medge, Mscrew=mscrew, vmax=100.0)
    #calforce = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=32, cell=net.cell)
    #calforce = CalForce(force_mode='LineTension', state=state, Ec=1.0e6)
    #calforce  = CalForce(force_mode='CUTOFF_MODEL', state=state, cutoff=0.2*Lbox)
    #mobility = MobilityLaw(mobility_law='FCC_0', state=state, Medge=65000.0, Mscrew=56000.0, vmax=100.0)
    #mobility  = MobilityLaw(mobility_law='SimpleGlide', state=state)
    #timeint   = TimeIntegration(integrator='EulerForward', dt=1e-9, state=state)
    timeint = TimeIntegration(integrator='Trapezoid', state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh = Remesh(remesh_rule='LengthBased', state=state)

    # 施加外部传入的应力 (注意这里移除了原本写死的 max_step=60000)
    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint, 
                          collision=collision, topology=topology, remesh=remesh, vis=None,
                          state=state, loading_mode='stress',
                          applied_stress=np.array([0.0, 0.0, 0.0, stress_value, 0, 0.0]),
                          print_freq=200, plot_freq=0, write_freq=200, write_dir=output_dir)
    
    # ==========================================
    # --- 优化版：分段模拟与动态裁判系统 (拖尾判定法) ---
    # ==========================================
    total_steps = 300000
    check_interval = 600
    passed = False

    # 提前算出 Loop 的理论中心位置 (用于划定过滤区域)
    loop_cx = Lbox * 0.5 - dist
    loop_cy = Lbox * 0.5
    loop_cz = Lbox * 0.5
    
    # 设定 Loop 的势力范围缓冲值 (稍微大于 radius)
    margin = 20.0 

    for current_step in range(0, total_steps, check_interval):
        # 1. 设定当前小目标步数，并让模拟器继续跑
        target_step = current_step + check_interval
        sim.max_step = target_step
        sim.run(net, state)

        # 2. 暂停获取当前节点信息
        try:
            from pyexadis_base import ExaDisNet
            G = net.get_disnet(ExaDisNet)
            
            nodes = G.get_positions()
            
            # --- 【步骤A：利用 Burgers Vector 提取纯 Line 节点】 ---
            # --- 【步骤A：利用 Burgers Vector 提取纯 Line 节点】 ---
            segs_data = G.get_segs_data() 
            b_loop_ref = b_prism 
            
            line_node_indices = set()
            loop_node_indices = set()
            
            # 从字典中提取节点对和柏氏矢量矩阵
            nodeids_array = segs_data['nodeids']
            burgers_array = segs_data['burgers']
            
            # 使用 enumerate 遍历所有的位错段
            for i, node_pair in enumerate(nodeids_array):
                n1, n2 = int(node_pair[0]), int(node_pair[1])
                b_seg = burgers_array[i] # 提取对应的 [bx, by, bz]
                
                # 【计算差异】：同时计算正向和反向的差异
                diff_plus = np.linalg.norm(b_seg - b_loop_ref)
                diff_minus = np.linalg.norm(b_seg + b_loop_ref)
                
                # 【判断】：只有既不是正向 Loop 也不是反向 Loop 的段，才被认定为 Line
                if diff_plus > 1e-5 and diff_minus > 1e-5:
                    line_node_indices.add(n1)
                    line_node_indices.add(n2)
                else:
                    loop_node_indices.add(n1)
                    loop_node_indices.add(n2)
            
            
            # 纯 Line 节点 = 属于 Line 段的节点，且排除掉交界处的节点 (可选)
            # 为了寻找拖尾，只要包含 line 柏氏矢量成分的节点都算作 line 节点
            pure_line_indices = list(line_node_indices) 
            
            if len(pure_line_indices) > 0:
                line_nodes = nodes[pure_line_indices]
                line_x_coords = line_nodes[:, 0]
                
                # --- 【步骤B：PBC 免疫的相对距离计算】 ---
                # 将绝对坐标系转换为以 Loop 为中心 [-Lbox/2, Lbox/2] 的相对坐标系
                # 公式： dx = (X_node - X_loop + L/2) % L - L/2
                dx_pbc = (line_x_coords - loop_cx + Lbox * 0.5) % Lbox - Lbox * 0.5
                
                # 寻找拖尾：因为整体向左 (-X) 运动，拖尾就是相对坐标最大的那个点
                trailing_edge_dx = np.max(dx_pbc)
                
                print(f"DEBUG: Step {target_step} | 拖尾相对 Loop 的距离 = {trailing_edge_dx:.2f} (负数代表在左，正数代表在右)")
                
                # --- 【步骤C：裁判逻辑】 ---
                # 要求最拖后的尾巴都在 Loop 左侧，并且留出 margin 的安全距离
                if trailing_edge_dx < -margin:
                    print(f"OUTCOME: PASS (Early stop at step {target_step}. Dislocation completely unpinned!)")
                    passed = True
                    break
            else:
                print(f"DEBUG: Step {target_step} | 体系中未找到 Line 节点。")
                
        #except Exception as e:
            #print(f"解析节点或段数据时发生异常: {e}")
            #sys.exit(1)
        except Exception as e:
            import traceback
            print("\n" + "="*40)
            print("🚨 裁判系统发生严重异常！详细报错如下：")
            traceback.print_exc()
            print("="*40 + "\n")
            sys.exit(1)

    # 4. 如果 500000 步全跑完了还没有达标，则判定为 FAIL（被钉扎住了）
    if not passed:
        print("OUTCOME: FAIL (Dislocation is pinned at the loop)")


if __name__ == "__main__":
    pyexadis.initialize() 

    # 接收从管家传来的参数，新增 --medge 和 --mscrew
    parser = argparse.ArgumentParser(description="Worker script for Single Simulation")
    parser.add_argument("--dist", type=float, default=50.0, help="Distance parameter")
    parser.add_argument("--radius", type=float, required=True, help="Radius of the loop")
    parser.add_argument("--stress", type=float, required=True, help="Applied stress in Pa")
    parser.add_argument("--medge", type=float, required=True, help="Edge Mobility")
    parser.add_argument("--mscrew", type=float, required=True, help="Screw Mobility")
    parser.add_argument("--id", type=int, default=1, help="Simulation Run ID")
    args = parser.parse_args()
    
    run_simulation(args.dist, args.radius, args.stress, args.medge, args.mscrew, args.id)
    
    pyexadis.finalize()    
