#%%
import xarray as xr
import numpy as np
import dynmodes as dm
import matplotlib.pyplot as plt
import gsw
import seaborn as sns
import pandas as pd
#%%
ds_t = xr.open_dataset('/mnt/d/data/woa13/woa13_decav_t00_01v2.nc',decode_times=False)
ds_s = xr.open_dataset('/mnt/d/data/woa13/woa13_decav_s00_01v2.nc',decode_times=False)
# %%
ds = xr.merge([ds_t["t_an"], ds_s["s_an"]])
lat = 25
lon = -165  # 195°E 等于 -165°W
temp = ds["t_an"].interp(lat=lat, lon=lon, method="linear")[0,:].values
salt = ds["s_an"].interp(lat=lat, lon=lon, method="linear")[0,:].values
depth_all = ds["depth"].values
pressure = gsw.p_from_z(-depth_all, lat)  # 从深度计算压力，注意深度为负值
SA = gsw.SA_from_SP(salt, pressure, lon, lat)  # 绝对盐度
CT = gsw.CT_from_t(SA, temp, pressure)  # 保守温度
sigma0 = gsw.sigma0(SA, CT)  # 参考压力为0 dbar的位密度
rho = sigma0 + 1000  # 转换为密度
N2_f = np.zeros_like(depth_all)
depth_mid = np.zeros_like(depth_all)
N2_f[1:], p_mid = gsw.Nsquared(SA, CT, pressure, lat=lat)
depth_mid[1:] = -gsw.z_from_p(p_mid, lat)  # 计算中间点的深度
N2_f[0] = N2_f[1]  # 处理边界条件
depth_mid[0] = 0  # 处理边界条件
#%%
#exclude the nan values
N2 = N2_f[~np.isnan(N2_f)]
depth_mid = depth_mid[~np.isnan(N2_f)]
depth = depth_all[~np.isnan(N2_f)]

# %%
wmodes, pmodes, ce,z,zp = dm.dynmodes(N2, -depth, nmodes=3,boundary='rigid',method='direct')

# %%
for i in range(pmodes.shape[1]):
    # 计算每个模态的最大绝对值
    max_abs = np.max(np.abs(pmodes[:, i]))
    # 归一化处理
    pmodes[:, i] = pmodes[:, i] / max_abs

for i in range(wmodes.shape[1]):
    # 计算每个模态的最大绝对值
    max_abs = np.max(np.abs(wmodes[:, i]))
    # 归一化处理
    wmodes[:, i] = wmodes[:, i] / max_abs

df_pmodes = pd.DataFrame({
    'Mode 1': pmodes[:,0],
    'Mode 2': pmodes[:,1],
    'Mode 3': pmodes[:,2],
    'depth': zp
})
df_wmodes = pd.DataFrame({
    'Mode 1': wmodes[:,0],
    'Mode 2': wmodes[:,1],
    'Mode 3': wmodes[:,2],
    'depth': z
})

sns.set_theme(style="whitegrid", context="poster")
fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True)

# 设置模态的颜色
mode_colors = {
    'Mode 1': '#6495ED',  # 浅蓝色
    'Mode 2': '#FA8072',  # 浅红色
    'Mode 3': '#FFB347'   # 浅橙色
}

# 绘制N²图像，使用黑色线条
axes[0].plot(np.sqrt(N2)*3600/(2*np.pi), -depth_mid, color='black')
axes[0].set_xticks([0, 2, 4, 6])
axes[0].set_xlim(left=0)
axes[0].set_ylim(bottom=-5000, top=0)
axes[0].set_ylabel('Depth (m)')
axes[0].set_xlabel('N (cph)')
axes[0].set_title('Buoyancy Frequency')
axes[0].grid(alpha=0.3)
mode_lines = []
# 绘制垂直速度模态图像
for i, mode in enumerate(['Mode 1', 'Mode 2', 'Mode 3']):
    mode_data = df_wmodes[mode].values
    line, = axes[1].plot(mode_data, z, label=mode, color=mode_colors[mode])
    mode_lines.append(line)
axes[1].axvline(x=0, color='grey', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Amplitude')
axes[1].set_title('Vertical Velocity Modes')
axes[1].grid(alpha=0.3)
wmode_xlim = axes[1].get_xlim()
# 绘制水平速度模态图像（包含图例）
for i, mode in enumerate(['Mode 1', 'Mode 2', 'Mode 3']):
    mode_data = df_pmodes[mode].values
    axes[2].plot(mode_data, zp, label=mode, color=mode_colors[mode])
axes[2].axvline(x=0, color='grey', linestyle='--', alpha=0.7)
axes[2].set_xlabel('Amplitude')
axes[2].set_title('Horizontal Velocity Modes')
axes[2].grid(alpha=0.3)
axes[2].set_xlim(wmode_xlim)
# axes[2].legend(loc='lower right')
axes[0].legend(mode_lines, ['Mode 1', 'Mode 2', 'Mode 3'], loc='lower right')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
# plt.savefig(f"dynmodes_lat{lat}_lon{-lon}.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
# if boundary == 'free', set the the nmodes = nmodes + 1
wmodes, pmodes, ce,z,zp = dm.dynmodes(N2, -depth, 4,'free')
# %%
for i in range(pmodes.shape[1]):
    # 计算每个模态的最大绝对值
    max_abs = np.max(np.abs(pmodes[:, i]))
    # 归一化处理
    pmodes[:, i] = pmodes[:, i] / max_abs
for i in range(wmodes.shape[1]):
    # 计算每个模态的最大绝对值
    max_abs = np.max(np.abs(wmodes[:, i]))
    # 归一化处理
    wmodes[:, i] = wmodes[:, i] / max_abs
df_pmodes = pd.DataFrame({
    'Mode 1': pmodes[:,0],
    'Mode 2': pmodes[:,1],
    'Mode 3': pmodes[:,2],
    'Mode 4': pmodes[:,3],
    'depth': zp
})
df_wmodes = pd.DataFrame({
    'Mode 1': wmodes[:,0],
    'Mode 2': wmodes[:,1],
    'Mode 3': wmodes[:,2],
    'Mode 4': wmodes[:,3],
    'depth': z
})
sns.set_theme(style="whitegrid", context="poster")
fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
# 设置模态的颜色
mode_colors = {
    'Mode 1': '#8A2BE2',  # 浅蓝色
    'Mode 2': '#6495ED',  # 浅红色
    'Mode 3': '#FA8072',  # 浅橙色
    'Mode 4': '#FFB347'   # 浅紫色
}
# 绘制N²图像，使用黑色线条
axes[0].plot(np.sqrt(N2)*3600/(2*np.pi), -depth_mid, color='black')
axes[0].set_xticks([0, 2, 4, 6])
axes[0].set_xlim(left=0)
axes[0].set_ylim(bottom=-5000, top=0)
axes[0].set_ylabel('Depth (m)')
axes[0].set_xlabel('N (cph)')
axes[0].set_title('Buoyancy Frequency')
axes[0].grid(alpha=0.3)
mode_lines = []
# 绘制垂直速度模态图像
for i, mode in enumerate(['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']):
    mode_data = df_wmodes[mode].values
    line, = axes[1].plot(mode_data, z, label=mode, color=mode_colors[mode])
    mode_lines.append(line)
axes[1].axvline(x=0, color='grey', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Amplitude')
axes[1].set_title('Vertical Velocity Modes')
axes[1].grid(alpha=0.3)
wmode_xlim = axes[1].get_xlim()
# 绘制水平速度模态图像（包含图例）
for i, mode in enumerate(['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']):
    mode_data = df_pmodes[mode].values
    axes[2].plot(mode_data, zp, label=mode, color=mode_colors[mode])
axes[2].axvline(x=0, color='grey', linestyle='--', alpha=0.7)
axes[2].set_xlabel('Amplitude')
axes[2].set_title('Horizontal Velocity Modes')
axes[2].grid(alpha=0.3)
axes[2].set_xlim(wmode_xlim)
# axes[2].legend(loc='lower right')
axes[0].legend(mode_lines, ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4'], loc='lower right')
plt.tight_layout()
plt.subplots_adjust(top=0.93)
# plt.savefig(f"dynmodes_lat{lat}_lon{-lon}_free.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
