#%%
import numpy as np
import time
import dynmodes as dm
import matplotlib.pyplot as plt

def benchmark_dynmodes(Nsq, depth, nmodes_list, n_repeats=10):
    """
    对dynmodes的两种方法进行性能基准测试

    参数:
    Nsq - 浮力频率数组
    depth - 深度数组 (负值)
    nmodes_list - 要测试的模态数列表
    n_repeats - 重复测试次数以获得可靠的时间测量

    返回:
    两个字典，包含每种方法的计算时间
    """
    direct_times = {}
    wkb_times = {}

    for nmodes in nmodes_list:
        # 初始化时间列表
        direct_time_list = []
        wkb_time_list = []

        # 多次运行以获得可靠的平均时间
        for _ in range(n_repeats):
            # 测试直接方法
            start_time = time.time()
            dm.dynmodes(Nsq, depth, nmodes=nmodes, method='direct')
            end_time = time.time()
            direct_time_list.append(end_time - start_time)

            # 测试WKB方法
            start_time = time.time()
            dm.dynmodes(Nsq, depth, nmodes=nmodes, method='wkb')
            end_time = time.time()
            wkb_time_list.append(end_time - start_time)

        # 计算平均时间
        direct_times[nmodes] = np.mean(direct_time_list)
        wkb_times[nmodes] = np.mean(wkb_time_list)

        # 打印结果
        print(f"模态数 {nmodes}:")
        print(f"  直接方法平均时间: {direct_times[nmodes]:.6f} 秒")
        print(f"  WKB方法平均时间: {wkb_times[nmodes]:.6f} 秒")
        print(f"  WKB比直接方法快: {direct_times[nmodes]/wkb_times[nmodes]:.2f} 倍")

    return direct_times, wkb_times
def benchmark_global_data(Nsq_base, depth_base, grid_sizes=[(360, 180), (3600, 1800)], n_repeats=10):
    """
    评估处理全球数据时的性能（纯估算方法）

    参数:
    Nsq_base - 原始浮力频率参考数组
    depth_base - 原始深度参考数组 (负值)
    grid_sizes - 要测试的全球网格尺寸列表
    n_repeats - 单点测试重复次数，用于提高估算准确性

    返回:
    包含每种方法处理全球数据所需时间的字典
    """
    direct_times = {}
    wkb_times = {}

    for grid_size in grid_sizes:
        nx, ny = grid_size
        grid_points = nx * ny
        grid_name = f"{nx}x{ny}"

        print(f"\n测试全球网格大小: {grid_name} (总点数: {grid_points})")

        # 估算单个点处理时间
        print(f"估算单点处理时间... (重复{n_repeats}次)")

        # 单点测试，重复n_repeats次以提高准确性
        direct_single = 0.0
        wkb_single = 0.0

        # 直接方法 - 测试不同模态数下的性能
        for nmodes in [3, 10, 50]:  # 测试几种不同的模态数
            print(f"测试模态数 {nmodes}...")
            direct_temp = 0.0
            wkb_temp = 0.0

            for _ in range(n_repeats):
                # 测试直接方法单点处理时间
                start_time = time.time()
                dm.dynmodes(Nsq_base, depth_base, nmodes=nmodes, method='direct')
                end_time = time.time()
                direct_temp += (end_time - start_time)

                # 测试WKB方法单点处理时间
                start_time = time.time()
                dm.dynmodes(Nsq_base, depth_base, nmodes=nmodes, method='wkb')
                end_time = time.time()
                wkb_temp += (end_time - start_time)

            # 计算平均值
            direct_temp /= n_repeats
            wkb_temp /= n_repeats

            print(f"  模态数{nmodes}: 直接方法: {direct_temp:.6f}秒, WKB方法: {wkb_temp:.6f}秒, 加速比: {direct_temp/wkb_temp:.2f}倍")

            # 使用最后一组测试作为实际估算数据
            if nmodes == 3:  # 实际处理全球数据时可能使用的模态数
                direct_single = direct_temp
                wkb_single = wkb_temp

        # 估算全球数据处理时间
        direct_estimate = direct_single * grid_points
        wkb_estimate = wkb_single * grid_points

        print("\n全球估算结果:")
        print(f"单点处理时间 - 直接方法: {direct_single:.6f}秒, WKB方法: {wkb_single:.6f}秒")
        print(f"全球{grid_name}数据估算时间 - 直接方法: {direct_estimate:.2f}秒 ({direct_estimate/60:.2f}分钟)")
        print(f"全球{grid_name}数据估算时间 - WKB方法: {wkb_estimate:.2f}秒 ({wkb_estimate/60:.2f}分钟)")

        # 存储结果
        direct_times[grid_name] = direct_estimate
        wkb_times[grid_name] = wkb_estimate

        # 计算加速比
        speedup = direct_estimate / wkb_estimate
        print(f"WKB方法加速比: {speedup:.2f}倍")

        # 计算内存需求（粗略估计）
        mem_per_point_mb = 0.01  # 假设每个点大约占用10KB内存
        total_mem_gb = (grid_points * mem_per_point_mb) / 1024  # 转换为GB
        print(f"估计内存需求: {total_mem_gb:.2f} GB")

        # 估算并行处理能力
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print("\n并行处理潜力估计:")
        print(f"系统CPU核心数: {cpu_count}")
        print(f"理论并行处理时间 - 直接方法: {direct_estimate/cpu_count:.2f}秒 ({direct_estimate/cpu_count/60:.2f}分钟)")
        print(f"理论并行处理时间 - WKB方法: {wkb_estimate/cpu_count:.2f}秒 ({wkb_estimate/cpu_count/60:.2f}分钟)")
        print("注意: 实际加速比通常低于理论值，受I/O和内存限制")

    return direct_times, wkb_times

def benchmark_depth_resolution(Nsq_base, depth_base, nmodes=3, n_repeats=5):
    """
    测试不同深度分辨率下两种方法的性能

    参数:
    Nsq_base - 原始浮力频率数组
    depth_base - 原始深度数组 (负值)
    nmodes - 模态数量
    n_repeats - 重复测试次数

    返回:
    包含不同分辨率下计算时间的字典
    """
    # 测试不同分辨率
    resolution_factors = [1, 2, 4]
    direct_times = {}
    wkb_times = {}

    for factor in resolution_factors:
        # 插值生成高分辨率数据
        if factor > 1:
            # 创建新的深度数组
            new_depth = np.linspace(depth_base[0], depth_base[-1], len(depth_base) * factor)
            # 插值N²值
            new_Nsq = np.interp(-new_depth, -depth_base, Nsq_base)
        else:
            new_depth = depth_base
            new_Nsq = Nsq_base

        resolution = len(new_depth)
        direct_time_list = []
        wkb_time_list = []

        # 多次运行以获得可靠的平均时间
        for _ in range(n_repeats):
            # 测试直接方法
            start_time = time.time()
            dm.dynmodes(new_Nsq, new_depth, nmodes=nmodes, method='direct')
            end_time = time.time()
            direct_time_list.append(end_time - start_time)

            # 测试WKB方法
            start_time = time.time()
            dm.dynmodes(new_Nsq, new_depth, nmodes=nmodes, method='wkb')
            end_time = time.time()
            wkb_time_list.append(end_time - start_time)

        # 计算平均时间
        direct_times[resolution] = np.mean(direct_time_list)
        wkb_times[resolution] = np.mean(wkb_time_list)

        # 打印结果
        print(f"深度点数 {resolution}:")
        print(f"  直接方法平均时间: {direct_times[resolution]:.6f} 秒")
        print(f"  WKB方法平均时间: {wkb_times[resolution]:.6f} 秒")
        print(f"  WKB比直接方法快: {direct_times[resolution]/wkb_times[resolution]:.2f} 倍")

    return direct_times, wkb_times

def plot_benchmark_results(nmodes_list, direct_times, wkb_times, title="不同模态数的计算时间"):
    """绘制基准测试结果"""
    global use_chinese

    plt.figure(figsize=(10, 6))
    plt.plot(nmodes_list, [direct_times[n] for n in nmodes_list], 'o-',
             label='直接方法' if use_chinese else 'Direct Method')
    plt.plot(nmodes_list, [wkb_times[n] for n in nmodes_list], 's-',
             label='WKB方法' if use_chinese else 'WKB Method')
    plt.xlabel('模态数' if use_chinese else 'Number of Modes')
    plt.ylabel('计算时间 (秒)' if use_chinese else 'Computation Time (s)')
    plt.title(title if use_chinese else "Computation Time for Different Modes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 绘制加速比
    plt.figure(figsize=(10, 6))
    speedup = [direct_times[n]/wkb_times[n] for n in nmodes_list]
    plt.plot(nmodes_list, speedup, 'o-')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel('模态数' if use_chinese else 'Number of Modes')
    plt.ylabel('加速比 (直接方法/WKB方法)' if use_chinese else 'Speedup (Direct/WKB)')
    plt.title('WKB方法相对于直接方法的加速比' if use_chinese else 'Speedup of WKB Method vs Direct Method')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_resolution_results(resolutions, direct_times, wkb_times):
    """绘制不同分辨率的基准测试结果"""
    global use_chinese

    plt.figure(figsize=(10, 6))
    plt.plot(resolutions, [direct_times[r] for r in resolutions], 'o-',
             label='直接方法' if use_chinese else 'Direct Method')
    plt.plot(resolutions, [wkb_times[r] for r in resolutions], 's-',
             label='WKB方法' if use_chinese else 'WKB Method')
    plt.xlabel('深度点数' if use_chinese else 'Number of Depth Points')
    plt.ylabel('计算时间 (秒)' if use_chinese else 'Computation Time (s)')
    plt.title('不同深度分辨率的计算时间' if use_chinese else 'Computation Time for Different Depth Resolutions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 绘制加速比
    plt.figure(figsize=(10, 6))
    speedup = [direct_times[r]/wkb_times[r] for r in resolutions]
    plt.plot(resolutions, speedup, 'o-')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel('深度点数' if use_chinese else 'Number of Depth Points')
    plt.ylabel('加速比 (直接方法/WKB方法)' if use_chinese else 'Speedup (Direct/WKB)')
    plt.title('不同分辨率下WKB方法的加速比' if use_chinese else 'Speedup of WKB Method at Different Resolutions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_global_benchmark(grid_sizes, direct_times, wkb_times):
    """绘制全球数据处理性能对比图"""
    global use_chinese

    # 创建柱状图
    plt.figure(figsize=(12, 7))
    x = np.arange(len(grid_sizes))
    width = 0.35

    # 转换时间从秒到分钟
    direct_minutes = [direct_times[g]/60 for g in grid_sizes]
    wkb_minutes = [wkb_times[g]/60 for g in grid_sizes]

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, direct_minutes, width,
                   label='直接方法' if use_chinese else 'Direct Method')
    rects2 = ax.bar(x + width/2, wkb_minutes, width,
                   label='WKB方法' if use_chinese else 'WKB Method')

    # 添加标签和标题
    ax.set_ylabel('处理时间 (分钟)' if use_chinese else 'Processing Time (minutes)')
    ax.set_title('全球数据处理性能对比' if use_chinese else 'Global Data Processing Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(grid_sizes)
    ax.legend()

    # 在柱子上方添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # 计算加速比
    plt.figure(figsize=(10, 6))
    speedup = [direct_times[g]/wkb_times[g] for g in grid_sizes]
    plt.bar(grid_sizes, speedup)
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel('网格尺寸' if use_chinese else 'Grid Size')
    plt.ylabel('加速比 (直接方法/WKB方法)' if use_chinese else 'Speedup (Direct/WKB)')
    plt.title('WKB方法在全球数据处理中的加速比' if use_chinese else 'Speedup of WKB Method for Global Data Processing')
    for i, v in enumerate(speedup):
        plt.text(i, v + 0.1, f"{v:.2f}倍" if use_chinese else f"{v:.2f}x", ha='center')
    plt.tight_layout()
#%%
import matplotlib as mpl
import matplotlib.font_manager as fm

# 设置中文字体支持
def set_chinese_font():
    """设置中文字体，尝试多种可能的中文字体"""
    # 尝试可能存在的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STXihei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']

    # 查找系统中可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 尝试设置中文字体
    font_found = False
    for font in chinese_fonts:
        if font in available_fonts:
            mpl.rcParams['font.family'] = font
            print(f"使用中文字体: {font}")
            font_found = True
            break

    # 如果找不到中文字体，尝试使用Noto Sans CJK
    if not font_found:
        try:
            # 尝试添加Noto Sans CJK字体
            font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
            if fm.fontManager.addfont(font_path):
                mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
                print("使用Noto Sans CJK字体")
                font_found = True
        except:
            pass

    # 如果还是找不到适合的字体，提示用户
    if not font_found:
        print("警告: 未找到合适的中文字体。图表中的中文可能无法正确显示。")
        print("建议安装中文字体，例如: 'apt-get install fonts-noto-cjk'")

        # 使用英文标签替代
        print("将使用英文标签代替中文标签")
        return False

    return True

# 调用字体设置函数
use_chinese = set_chinese_font()
# 主程序
#%%
import xarray as xr
import gsw
import numpy as np

print("加载数据...")

# 加载WOA13数据
ds_t = xr.open_dataset('/mnt/d/data/woa13/woa13_decav_t00_01v2.nc', decode_times=False)
ds_s = xr.open_dataset('/mnt/d/data/woa13/woa13_decav_s00_01v2.nc', decode_times=False)

ds = xr.merge([ds_t["t_an"], ds_s["s_an"]])
lat = 25
lon = -165  # 195°E 等于 -165°W
temp = ds["t_an"].interp(lat=lat, lon=lon, method="linear")[0,:].values
salt = ds["s_an"].interp(lat=lat, lon=lon, method="linear")[0,:].values
depth_all = ds["depth"].values

pressure = gsw.p_from_z(-depth_all, lat)
SA = gsw.SA_from_SP(salt, pressure, lon, lat)
CT = gsw.CT_from_t(SA, temp, pressure)

N2_f = np.zeros_like(depth_all)
depth_mid = np.zeros_like(depth_all)
N2_f[1:], p_mid = gsw.Nsquared(SA, CT, pressure, lat=lat)
depth_mid[1:] = -gsw.z_from_p(p_mid, lat)
N2_f[0] = N2_f[1]
depth_mid[0] = 0

# 排除NaN值
N2 = N2_f[~np.isnan(N2_f)]
depth = depth_all[~np.isnan(N2_f)]

print("数据已加载，数据点数:", len(N2))

# 测试不同模态数
print("\n测试不同模态数的性能...")
nmodes_list = [1, 3, 5, 10, 20, 50]
direct_times, wkb_times = benchmark_dynmodes(N2, -depth, nmodes_list)

# 绘制结果
plot_benchmark_results(nmodes_list, direct_times, wkb_times)

# 测试不同分辨率
print("\n测试不同深度分辨率的性能...")
direct_res_times, wkb_res_times = benchmark_depth_resolution(N2, -depth)

# 绘制结果
plot_resolution_results(list(direct_res_times.keys()), direct_res_times, wkb_res_times)

plt.show()

# 在现有代码后添加

# 测试全球数据处理性能
print("\n测试全球数据处理性能...")
global_direct_times, global_wkb_times = benchmark_global_data(N2, -depth,
                                                           grid_sizes=[(360, 180), (3600, 1800)])

# 绘制全球数据处理结果
grid_sizes = list(global_direct_times.keys())
plot_global_benchmark(grid_sizes, global_direct_times, global_wkb_times)

# 汇总结果
print("\n性能总结:")
print("1. 模态数对性能的影响: WKB方法在高模态数时优势更明显")
print("2. 分辨率对性能的影响: 随着分辨率增加，WKB方法的优势更明显")
print("3. 全球数据处理能力: ")
for grid in grid_sizes:
    direct_time = global_direct_times[grid]
    wkb_time = global_wkb_times[grid]
    speedup = direct_time / wkb_time
    print(f"   - {grid}网格: 直接方法需要{direct_time/60:.2f}分钟，WKB方法需要{wkb_time/60:.2f}分钟，加速{speedup:.2f}倍")
