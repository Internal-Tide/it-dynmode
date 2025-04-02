import matplotlib
matplotlib.use('Agg') # 启用图形保存模式

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import scipy.stats as stats
import scipy.ndimage as nd
from netCDF4 import Dataset
import warnings
warnings.filterwarnings('ignore')
from mpi4py import MPI
import gsw as gsw
from datetime import datetime
# - 自定义函数 - 
from change_coord import reproject_image_into_polar
from distance_sphere_matproof import dist_sphere_matproof
from convert_TPXO_to_ellipses import ellipse
clock = datetime.now()

# --- MPI参数设置 --- 
# x和y方向的处理器数量
npx,npy = 2,1
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size!=npx*npy:
    exit('子域数量与处理器数量不匹配 -> 退出!')

# ------------ 参数设置 ------------
debug             = False # 详细输出模式
rho0              = 1025. # 海水密度 [kg m-3]
tide              = 'M2'  # 潮汐类型：M2, S2, K1 
topo              = 'srtm30' # 地形数据
dspace            = 1.  # 网格分辨率（度）
region            = 'globalhr' # 区域
lonmin_g,lonmax_g = -180,180 # 网格边界     
latmin_g,latmax_g = -76,76
nmodes            = 10           # 要保存的模态数量   
method_itp        = 'nearest'    # 地形处理的插值方法
zmin              = -500         # 计算Ef的最小深度[m]，大致是陆架以下
g                 = 9.81         # 重力加速度 [m s-2]
path_data         = '/Users/cv1m15/Data/'
file_woat         = [path_data+'woa18_decav_t%.2i_01.nc'%i for i in range(13)] # 年度+月度数据
file_woas         = [path_data+'woa18_decav_s%.2i_01.nc'%i for i in range(13)] 
file_woa_bathy    = path_data+'ETOPO2v2c_f4_woa.nc' # WOA网格上的ETOPO2地形数据

if debug: 
    file_output = path_data+'Ef_'+topo+'_'+tide+'_'+region+'_debug_%.3i.nc'%rank
else: 
    file_output = path_data+'Ef_'+topo+'_'+tide+'_'+region+'_%.3i.nc'%rank


# --- 依赖参数设置 --- 
if tide == 'M2':
    file_tpxo_u = path_data+'u_m2_tpxo8_atlas_30.nc'
    omega = 2.*np.pi/(44700.)   # 潮汐频率 [rad s-1]
elif tide == 'S2':
    file_tpxo_u = path_data+'u_s2_tpxo8_atlas_30.nc'
    omega = 2.*np.pi/(43200.)
elif tide == 'K1':
    file_tpxo_u = path_data+'u_k1_tpxo8_atlas_30.nc'
    omega = 2.*np.pi/(86164.)
file_tpxo_h = path_data+'grid_tpxo8_atlas.nc'
if topo == 'srtm30':
    file_topo = path_data+'srtm30-plus_global.nc'
    nextend   = 1600 
elif topo == 'etopo2':
    file_topo = path_data+'ETOPO2v2c_f4.nc'
    nextend   = 400 
    
# --- 根据处理器编号设置坐标 ---
lon1d_g       = np.arange(lonmin_g,lonmax_g+dspace,dspace)
lat1d_g       = np.arange(latmin_g,latmax_g+dspace,dspace)
nlat_g,nlon_g = lat1d_g.shape[0],lon1d_g.shape[0]
di            = nlon_g/npx
dj            = nlat_g/npy
if di*npx<nlon_g: di+=1 # 修正以确保覆盖整个区域
if dj*npy<nlat_g: dj+=1
imin          = int(di*(rank%npx))
jmin          = int(dj*(rank//npx))
imax          = int(imin+di)
jmax          = int(jmin+dj)

# --- 定义子网格 ---
lon1d = lon1d_g[imin:imax]
lat1d = lat1d_g[jmin:jmax]
lon1d_e = np.concatenate(([lon1d[0]-dspace/2.],0.5*(lon1d[1:]+lon1d[:-1]),[lon1d[-1]+dspace/2.])) # 边缘点
lat1d_e = np.concatenate(([lat1d[0]-dspace/2.],0.5*(lat1d[1:]+lat1d[:-1]),[lat1d[-1]+dspace/2.])) # 边缘点
lon2d,lat2d = np.meshgrid(lon1d,lat1d)
nlat,nlon = lon2d.shape
lonmin,lonmax = np.nanmin(lon2d),np.nanmax(lon2d)
latmin,latmax = np.nanmin(lat2d),np.nanmax(lat2d)

# --- 显示网格设置 --- 
lonall = comm.gather([lonmin,lonmax],root=0)
latall = comm.gather([latmin,latmax],root=0)
dimall = comm.gather([nlat,nlon],root=0)
if rank==0:
    print(' 总计算域：经度范围 [%i,%i], 纬度范围 [%i,%i]'%(lonmin_g,lonmax_g,latmin_g,latmax_g))
    for i in range(size):
        print('  处理器 %.3i 负责经度范围 [%.2f,%.2f], 纬度范围 [%.2f,%.2f], [%i x %i] 个点'\
              %(i,lonall[i][0],lonall[i][1],latall[i][0],latall[i][1],dimall[i][1],dimall[i][0]))
exit()
# ------------ 处理TPXO数据 ------------ 
if rank==0:
    print('####################################################################################')
    print('############# 处理TPXO流速数据 ################################################')
    print('####################################################################################')
# --- 注：所有变量都是(经度,纬度)格式 --- 
nc = Dataset(file_tpxo_u,'r') 
u_tmp = nc.variables['uRe'][:]+1j*nc.variables['uIm'][:] 
v_tmp = nc.variables['vRe'][:]+1j*nc.variables['vIm'][:] 
nc.close()
nc = Dataset(file_tpxo_h,'r') 
hu = nc.variables['hu'][:]
hv = nc.variables['hv'][:]
lon_u = nc.variables['lon_u'][:]; lon_u[lon_u>180]-=360
lat_u = nc.variables['lat_u'][:]
lon_v = nc.variables['lon_v'][:]; lon_v[lon_v>180]-=360
lat_v = nc.variables['lat_v'][:]
nc.close()
# --- 重新排列使经度范围为[-180,180]而非[0,360] ---
midx = int(u_tmp.shape[0]/2) 
u_tmp = np.concatenate((u_tmp[midx:,:],u_tmp[:midx,:]),axis=0)
v_tmp = np.concatenate((v_tmp[midx:,:],v_tmp[:midx,:]),axis=0)
hu    = np.concatenate((hu[midx:,:],hu[:midx,:]),axis=0)
hv    = np.concatenate((hv[midx:,:],hv[:midx,:]),axis=0)
lon_u = np.concatenate((lon_u[midx:],lon_u[:midx]))
lon_v = np.concatenate((lon_v[midx:],lon_v[:midx]))

# --- 为了后续分箱处理，将数据扩展到经度<-180和经度>180的区域 ---
lon_u = np.concatenate((lon_u[-20:]-360,lon_u,lon_u[:20]+360)) 
lon_v = np.concatenate((lon_v[-20:]-360,lon_v,lon_v[:20]+360)) 
u_tmp = np.concatenate((u_tmp[-20:,:],u_tmp,u_tmp[:20,:]),axis=0) 
v_tmp = np.concatenate((v_tmp[-20:,:],v_tmp,v_tmp[:20,:]),axis=0) 
hu    = np.concatenate((hu[-20:,:],hu,hu[:20,:]),axis=0)
hv    = np.concatenate((hv[-20:,:],hv,hv[:20,:]),axis=0) 

# --- 子采样以便进行分箱处理 --- 
imin_u = np.nanargmin(abs(lon_u - lonmin))-10
imax_u = np.nanargmin(abs(lon_u - lonmax))+10 
jmin_u = np.nanargmin(abs(lat_u - latmin))-10
jmax_u = np.nanargmin(abs(lat_u - latmax))+10 
imin_v = np.nanargmin(abs(lon_v - lonmin))-10
imax_v = np.nanargmin(abs(lon_v - lonmax))+10 
jmin_v = np.nanargmin(abs(lat_v - latmin))-10
jmax_v = np.nanargmin(abs(lat_v - latmax))+10
lon_u,lat_u = np.meshgrid(lon_u[imin_u:imax_u],lat_u[jmin_u:jmax_u]) 
lon_v,lat_v = np.meshgrid(lon_v[imin_v:imax_v],lat_v[jmin_v:jmax_v]) 
u_tmp       = u_tmp[imin_u:imax_u,jmin_u:jmax_u] 
v_tmp       = v_tmp[imin_v:imax_v,jmin_v:jmax_v] 
hu          =    hu[imin_u:imax_u,jmin_u:jmax_u] 
hv          =    hv[imin_v:imax_v,jmin_v:jmax_v] 

# --- 计算振幅和相位 --- 
ua_tmp = abs(u_tmp)/(hu*1e2) # [cm2/s]转换为[cm/s]
va_tmp = abs(v_tmp)/(hv*1e2) # [cm2/s]转换为[cm/s]
up_tmp = np.arctan2(-np.imag(u_tmp),np.real(u_tmp))/np.pi*180
vp_tmp = np.arctan2(-np.imag(v_tmp),np.real(v_tmp))/np.pi*180
up_tmp[up_tmp<0]+=360
vp_tmp[vp_tmp<0]+=360

# --- 分箱到局部网格 --- 
[ua,_,_,_] = stats.binned_statistic_2d(np.ravel(lat_u),np.ravel(lon_u),
                                       np.ravel(ua_tmp.T),statistic=np.nanmean,bins=[lat1d_e,lon1d_e]) 
[up,_,_,_] = stats.binned_statistic_2d(np.ravel(lat_u),np.ravel(lon_u),
                                       np.ravel(up_tmp.T),statistic=np.nanmean,bins=[lat1d_e,lon1d_e]) 
[va,_,_,_] = stats.binned_statistic_2d(np.ravel(lat_v),np.ravel(lon_v),
                                       np.ravel(va_tmp.T),statistic=np.nanmean,bins=[lat1d_e,lon1d_e]) 
[vp,_,_,_] = stats.binned_statistic_2d(np.ravel(lat_v),np.ravel(lon_v),
                                       np.ravel(vp_tmp.T),statistic=np.nanmean,bins=[lat1d_e,lon1d_e]) 

[sema,ecc,phi,pha] = ellipse(ua,up,va,vp) # phi是半长轴与x轴的夹角（度）
ue  = sema*1e-2      # 半长轴方向的速度 [m/s]
ve  = sema*ecc*1e-2  # 半短轴方向的速度 [m/s]
phi = phi*np.pi/180  # 转换为弧度
del(ua_tmp,va_tmp,up_tmp,vp_tmp,lon_u,lat_u,lon_v,lat_v) # 释放内存

# ------------ 处理地形数据 ------------ 
if rank==0:
    print('####################################################################################')
    print('############# 处理地形数据 ##############################################')
    print('####################################################################################')
nc = Dataset(file_topo,'r')
lonh_glo = nc.variables['x'][:]
lath_glo = nc.variables['y'][:]
h_glo    = nc.variables['z'][:]
nc.close()
res_topo = (lonh_glo[1]-lonh_glo[0])*60*1852 # [m] 分辨率

# --- 为插值目的将数据扩展到经度<-180和经度>180的区域 --- 
lonh_glo = np.concatenate((lonh_glo[-nextend-1:]-360,lonh_glo,lonh_glo[:nextend+1]+360))
h_glo    = np.concatenate((h_glo[:,-nextend-1:],h_glo,h_glo[:,:nextend+1]),axis=1)

# --- 提取子集以便进行插值操作 --- 
imin  = np.nanargmin(abs(lonh_glo - lonmin))-nextend
imax  = np.nanargmin(abs(lonh_glo - lonmax))+nextend
jmin  = np.nanargmin(abs(lath_glo - latmin))-nextend
jmax  = np.nanargmin(abs(lath_glo - latmax))+nextend
lonh  = lonh_glo[imin:imax+1]
lath  = lath_glo[jmin:jmax+1]
h_loc = h_glo[jmin:jmax+1,imin:imax+1] # 每个处理器的"局部"地形数据 
lonh2d,lath2d = np.meshgrid(lonh,lath) 
del(h_glo,lonh_glo,lath_glo) # 释放内存

# --- 将地形分箱到计算网格 --- 
[h_grid,_,_,_] = stats.binned_statistic_2d(np.ravel(lath2d),np.ravel(lonh2d),np.ravel(h_loc),
                                          statistic=np.nanmean,bins=[lat1d_e,lon1d_e]) 
h_grid[h_grid>0] = 0     # 陆地点设为0

# --- 平滑网格地形以便之后计算通量方向 --- 
n_smooth  = 1 # 需要根据主网格的分辨率调整
h_grid_lp = nd.filters.gaussian_filter(h_grid,n_smooth)
h_itp     = itp.RectBivariateSpline(lat1d,lon1d,h_grid_lp,kx=1,ky=1) # 线性插值

# ------------ 处理WOA数据 ------------ 
# 注：策略是在WOA网格上计算所有内容（var_woa），然后插值到局部网格
if rank==0:
    print('####################################################################################')
    print('############# 处理WOA数据 #####################################################')
    print('####################################################################################')
# --- 读取年度数据和地形数据 --- 
nc     = Dataset(file_woat[0],'r')
lat_woa = nc.variables['lat'][:]
lon_woa = nc.variables['lon'][:]
z      = -nc.variables['depth'][:]
kmin   = np.argwhere(z==-1500)[0][0]+1
t_tmp  = nc.variables['t_an'][0,:,:,:]
t_yr   = nc.variables['t_an'][0,kmin:,:,:]
nc.close()
nc     = Dataset(file_woas[0],'r')
s_yr   = nc.variables['s_an'][0,kmin:,:,:]
nc.close()
nc     = Dataset(file_woa_bathy,'r') 
h_woa  = nc.variables['h'][:] 
nc.close()   
h_woa[h_woa>0] = 0; h_woa = -h_woa  

# --- 读取月度数据 --- 
nzwoa,nywoa,nxwoa = t_tmp.shape
lon_woa,lat_woa   = np.meshgrid(lon_woa,lat_woa)
z_tile            = np.tile(np.tile(z,(nxwoa,1)),(nywoa,1,1)).transpose(2,0,1)
N2_woa            = np.zeros((12,nzwoa-1,nywoa,nxwoa))
for m in range(12):
    nc = Dataset(file_woat[m+1],'r')
    t  = np.squeeze(nc.variables['t_an'][:])
    nc.close()
    t  = np.concatenate((t,t_yr),axis=0)
    nc = Dataset(file_woas[m+1],'r')
    s  = np.squeeze(nc.variables['s_an'][:])
    nc.close()
    s  = np.concatenate((s,s_yr),axis=0)
    # - 按照MacDougall和Barker 2011的说明 - 
    p  = gsw.p_from_z(z_tile,lat_woa)
    SA = gsw.SA_from_SP(s,p,lon_woa,lat_woa)
    CT = gsw.CT_from_t(SA,t,p)
    [N2month,p_mid] = gsw.Nsquared(SA,CT,p,lat_woa) # [(rad s^-1)^2] 
    z_mid = gsw.z_from_p(p_mid.data,lat_woa)
    N2month[N2month<0] = 1e-8 # 最小的分层在非常深的海底
    N2_woa[m,:,:,:] = N2month 


# --- 计算模态1波长 --- 
dz       = abs(np.diff(z_tile,axis=0))
Nbar_woa = np.nansum(np.sqrt(N2_woa)*dz,axis=1)/h_woa   
k1_woa   = np.zeros((12,nywoa,nxwoa))
for i in range(12):
    k1_woa[i,:,:] = (np.pi/h_woa)*((omega**2-gsw.f(lat_woa)**2)/\
                                   (Nbar_woa[i,:,:]**2-omega**2))**0.5

# --- 计算底层分层 --- 
N2b_woa = np.zeros((12,nywoa,nxwoa))
for m in range(12): 
    for j in range(nywoa):
        for i in range(nxwoa):
            N2tmp = N2_woa[m,:,j,i]
            if N2tmp[~np.isnan(N2tmp)].shape[0]>0:
                kmax = np.nanargmax(np.sort(N2tmp))
                # - 平均最深的3个层次以稍微平滑一下或平均底部500米
                N2b_woa[m,j,i] = np.nanmean(N2tmp[kmax-2:kmax+1]) 

#if rank==0:
#    nc = Dataset('tmp.nc','w') 
#    nc.createDimension('y',nywoa) 
#    nc.createDimension('x',nxwoa) 
#    nc.createDimension('month',12) 
#    nc.createVariable('N2b','f',('month','y','x')) 
#    nc.createVariable('k1','f',('month','y','x')) 
#    nc.variables['N2b'][:] = N2b_woa
#    nc.variables['k1'][:] = k1_woa
#    nc.close() 

# --- 现在插值到局部网格 --- 
k1  = np.zeros((12,nlat,nlon))
N2b = np.zeros((12,nlat,nlon))
N2b_woa[np.isnan(N2b_woa)] = 0 # 插值时没有nan
k1_woa[np.isnan(k1_woa)] = 0; k1_woa[k1_woa<0] = 0 # 临界纬度  
for m in range(12): 
    spline    = itp.RectBivariateSpline(lat_woa[:,0],lon_woa[0,:],k1_woa[m,:,:],kx=1,ky=1) 
    k1[m,:,:] = spline(lat1d,lon1d) 
    spline    = itp.RectBivariateSpline(lat_woa[:,0],lon_woa[0,:],N2b_woa[m,:,:],kx=1,ky=1) 
    N2b[m,:,:] = spline(lat1d,lon1d) 

lambda1 = 2*np.pi/np.nanmean(k1,axis=0) # [m] 年平均模态1波长 
lambda1_max = 250000 # [m] 最大波长以避免在临界纬度处出现奇点
lambda1[lambda1>lambda1_max] = lambda1_max

# ------------ 循环计算域 ------------ 
if rank==0:
    print('####################################################################################')
    print('############# 循环计算域 #######################################################')
    print('####################################################################################')
# - 要保存的变量 - 
npts = np.zeros((nlat,nlon)).astype(int)
if debug: # 保存水平波数和Efa 
    npts_max = int((np.sqrt(2)/2)*2*lambda1_max/(res_topo*np.cos(75*np.pi/180))) # max(2*wavelength)/min(resolution)  
    print('npts_max=%.i'%npts_max) 
    Efa    = np.zeros((12,nlat,nlon,npts_max)) # 方位积分能量通量  
    kk     = np.zeros((nlat,nlon,npts_max)) # 波数                          
Eft    = np.zeros((12,nlat,nlon)) 
Eft_sc = np.zeros((12,nlat,nlon)) 
Efn    = np.zeros((12,nmodes,nlat,nlon)) # 模态能量通量 
Efn_sc = np.zeros((12,nmodes,nlat,nlon)) # 模态能量通量 
gamma_sup_avg  = np.zeros((12,nlat,nlon)) # 平均gamma值，其中gamma>1 
gamma_sup_frac = np.zeros((12,nlat,nlon)) # 超临界坡度的比例 
theta_max = np.zeros((12,nmodes,nlat,nlon)) # 模态能量通量 
for j in range(nlat):
    clock_diff = datetime.now() - clock
    hour,sec = divmod(clock_diff.seconds,3600)
    hour     = hour + clock_diff.days*24
    minu,sec = divmod(sec,60)
    print(' ---> 处理器 %.3i, 已用时间 : %.2i h %.2i min %.2i sec, 计算进度 %.1f percent'\
          %(rank,hour,minu,sec,float(j)/nlat*100.)) 
    for i in range(nlon):
        if (h_grid[j,i]<zmin) and (omega>gsw.f(lat1d[j])) and (lambda1[j,i]>3*res_topo):
            # --- 获取地形块大小 --- 
            res_topo_x = res_topo*np.cos(lat1d[j]*np.pi/180) # x方向的地形分辨率      
            # 注：'0.5': 半宽，'2': 我们希望有2个波长 
            nn       = int(0.5*2*lambda1[j,i]/res_topo_x) 
            # --- 提取地形块 --- 
            if debug: print('      -> 处理器 %.3i, 提取地形块, nn=%i,lambda1=%i,res_topo_x=%.3f'\
                            %(rank,nn,lambda1[j,i],res_topo_x))
            ilon = np.nanargmin(abs(lonh - lon1d[i]))
            ilat = np.nanargmin(abs(lath - lat1d[j]))
            lon  = lonh[ilon-nn:ilon+nn]
            lat  = lath[ilat-nn:ilat+nn]
            h    = h_loc[ilat-nn:ilat+nn,ilon-nn:ilon+nn]
            h[h>0] = 0 
            # --- 计算局部网格度量 ---   
            lon,lat = np.meshgrid(lon,lat)
            xx = dist_sphere_matproof(lat,lon,lat,lon1d[i])
            yy = dist_sphere_matproof(lat,lon,lat1d[j],lon)
            xx[lon<lon1d[i]] = -xx[lon<lon1d[i]]
            yy[lat<lat1d[j]] = -yy[lat<lat1d[j]]
            # --- 现在设置一个规则网格 --- 
            if debug: print('      -> 处理器 %.3i, 设置规则网格'%rank)
            xi    = np.arange(np.nanmin(xx),np.nanmax(xx),res_topo_x)
            yi    = np.arange(np.nanmin(yy),np.nanmax(yy),res_topo_x)
            xi,yi = np.meshgrid(xi,yi)
            hi    = itp.griddata((np.ravel(xx),np.ravel(yy)),np.ravel(h),
                                 (xi,yi),method=method_itp)     
            #if rank==0: 
            #    plt.figure()
            #    plt.subplot(121);plt.contourf(xx,yy,h,20);plt.colorbar()  
            #    plt.subplot(122);plt.contourf(xi,yi,hi,20);plt.colorbar()  
            #    plt.savefig('tmp.png') 
            #exit()
      
            # --- 现在将网格旋转到主轴方向 --- 
            if debug: print('      -> 处理器 %.3i, 旋转网格'%rank)
            #print(' 旋转角度 [度，逆时针从x轴开始] ',phi[j,i]*180/np.pi) 
            xr    = xi*np.cos(phi[j,i]) - yi*np.sin(phi[j,i])
            yr    = xi*np.sin(phi[j,i]) + yi*np.cos(phi[j,i])
            hr    = itp.griddata((np.ravel(xi),np.ravel(yi)),np.ravel(hi),
                                 (xr,yr),method=method_itp)

            # --- 重塑以获得正方形并去除由于插值导致的边缘处的nan ---  
            midy,midx = int(hr.shape[0]/2),int(hr.shape[1]/2)
            nn_new = int(nn*np.sqrt(2)/2) # 保留的点数的一半 
            hr_sub = hr[midy-nn_new:midy+nn_new,midx-nn_new:midy+nn_new]
            xr_sub = xr[midy-nn_new:midy+nn_new,midx-nn_new:midy+nn_new]
            yr_sub = yr[midy-nn_new:midy+nn_new,midx-nn_new:midy+nn_new]
            npts[j,i] = hr_sub.shape[0] 
            # --- 计算地形梯度以检查超临界性 --- 
            dhdx   = np.diff(hr_sub,axis=1)/res_topo_x
            dhdy   = np.diff(hr_sub,axis=0)/res_topo_x
            dhdx_r = 0.5*(dhdx[:,1:]+dhdx[:,:-1])
            dhdy_r = 0.5*(dhdy[1:,:]+dhdy[:-1,:])
            h_grad = (dhdx_r[1:-1,:]**2 + dhdy_r[:,1:-1]**2)**0.5

            # --- 计算2D谱 --- 
            if debug: print('      -> 处理器 %.3i, 计算谱'%rank)
            nx = npts[j,i]  
            kx = np.fft.fftshift(np.fft.fftfreq(nx,res_topo_x))*2*np.pi # x方向的波数 
            ky = np.fft.fftshift(np.fft.fftfreq(nx,res_topo_x))*2*np.pi # y方向的波数 
            dkx = kx[1]-kx[0]
            win_x   = np.tile(np.hanning(nx),(1,1))  # 滤波前的窗口
            win_y   = np.tile(np.hanning(nx),(1,1)).T
            win     = np.dot(win_y,win_x) 
            hr_win  = (hr_sub - np.nanmean(hr_sub))*win # 去除均值并对信号加窗 
            sp      = abs(np.fft.fftshift(np.fft.fft2(hr_win)))**2
            # - 归一化 - 
            #norm_coef = np.nanvar(hr_sub) 
            midy,midx = int(hr_sub.shape[0]/2),int(hr_sub.shape[1]/2) 
            quarter   = int(hr_sub.shape[0]/4)
            # 归一化为lambda1*lambda1区域内的var(h) 
            norm_coef = np.nanvar(hr_sub[midy-quarter:midy+quarter,midx-quarter:midx+quarter]) 
            sp        = sp*norm_coef/np.sum(sp*dkx*dkx) 
            # --- 转换为极谱 --- 
            sp_polar, r, theta = reproject_image_into_polar(sp) # sp_polar(kh,theta) 
            kh = r*dkx # r是像素，乘以dkx得到波数
            dkh = kh[1]   
            # --- 用潮汐相关系数加权谱 --- 
            weight = ( ue[j,i]**2*np.cos(theta)**2
                     + ve[j,i]**2*np.sin(theta)**2 )
            for k in np.arange(nx):
                sp_polar[k,:] = sp_polar[k,:]*weight
            for t in np.arange(nx):
                sp_polar[:,t] = sp_polar[:,t]*kh
            if debug: # 保存变量  
                kk[j,i,:nx] = kh
            # --- 循环月份 --- 
            for m in range(12): 
                # - 计算Ef(K,theta) - 
                coef = 0.5*rho0*((N2b[m,j,i]-omega**2)*(omega**2-gsw.f(lat2d[j,i])**2))**0.5/omega
                Ef = coef*sp_polar 
                # - 方位积分 [0,2pi] -
                dtheta = theta[1] - theta[0]
                Efa_tmp = np.zeros(nx)
                for k in range(nx):
                    Efa_tmp[k] = np.nansum(Ef[k,:]*kh[k]*dtheta)
                # - 在物理上合理的波数范围内积分 -
                dkj  = k1[m,j,i] # 在"模态波长"空间中    
                kmin = k1[m,j,i]-0.5*dkj 
                kmax = 2*np.pi/(2*res_topo_x) # 奈奎斯特  
                # - 旧方法 -
                #kmin_ind = np.nanargmin(abs(kh-kmin)) 
                #kmax_ind = np.nanargmin(abs(kh-kmax)) 
                #Eft[m,j,i] = np.nansum(Efa_tmp[kmin_ind:kmax_ind+1]*dkh) 
                # - 新方法但不太好 -
                #interp = itp.InterpolatedUnivariateSpline(kh,Efa_tmp,k=1) # 线性插值 
                #Eft[m,j,i] = interp.integral(kmin,kmax) 
                # - 新方法且正确 -
                interp = itp.interp1d(kh,Efa_tmp,bounds_error=False) # 线性插值 
                kint = np.linspace(kmin,kmax,2*nx)
                dkint = kint[1]-kint[0]   
                Eft[m,j,i] = np.nansum(interp(kint)*dkint) 
                if debug: # 保存变量     
                    Efa[m,j,i,:nx] = Efa_tmp
                # - 模态通量 -  
                for n in range(nmodes): 
                    kmin = (n+1)*k1[m,j,i] - 0.5*dkj  
                    kmax = (n+1)*k1[m,j,i] + 0.5*dkj  
                    #Efn[m,n,j,i] = interp.integral(kmin,kmax) 
                    kint = np.linspace(kmin,kmax,100)
                    dkint = kint[1]-kint[0]   
                    Efn[m,n,j,i] = np.nansum(interp(kint)*dkint) 
           
                # - 超临界坡度的修正 -
                beam  = ((omega**2-gsw.f(lat1d[j])**2)/(N2b[m,j,i]-omega**2))**0.5
                gamma = h_grad/beam 
                gamma_sup = np.copy(gamma); gamma_sup[gamma_sup<1] = np.nan 
                gamma_sup_avg[m,j,i]  = np.nanmean(gamma_sup)
                if np.isnan(gamma_sup_avg[m,j,i]):gamma_sup_avg[m,j,i] = 1 # 避免Eft_sc中的nan 
                gamma_sup_frac[m,j,i] = gamma_sup[~np.isnan(gamma_sup)].shape[0]/np.ravel(gamma).shape[0] 
                Eft_sc[m,j,i]    = Eft[m,j,i]*gamma_sup_frac[m,j,i]/gamma_sup_avg[m,j,i]**2\
                                 + Eft[m,j,i]*(1-gamma_sup_frac[m,j,i])
                Efn_sc[m,:,j,i]  = Efn[m,:,j,i]*gamma_sup_frac[m,j,i]/gamma_sup_avg[m,j,i]**2\
                                 + Efn[m,:,j,i]*(1-gamma_sup_frac[m,j,i])
                # - 获取最大通量的角度 - 
                ff = itp.interp1d(kh,Ef,axis=0,bounds_error=False)
                for n in range(nmodes):
                    Ef_dir = ff((n+1)*k1[m,j,i])*dkj
                    try:
                        ind_max = np.nanargmax(Ef_dir)
                    except:
                        ind_max = 0
                    thetam  = theta[ind_max]               # 在[-pi,pi]范围内 
                    if thetam < 0: thetam += 2*np.pi       # 在[0,2pi]范围内
                    thetam += phi[j,i]                     # 在[0,4pi]范围内 
                    if thetam > 2*np.pi: thetam -= 2*np.pi # 在[0,2pi]范围内 
                    if thetam > np.pi:   thetam -= np.pi   # 在[0,pi]范围内
                    # 获取thetam方向的地形梯度 
                    dl   = 0.5 # 经度、纬度的度数以插值地形 
                    dlon = dl*np.cos(thetam)
                    dlat = dl*np.sin(thetam)
                    lon0,lat0 = lon2d[j,i]+dlon, lat2d[j,i]+dlat
                    lon1,lat1 = lon2d[j,i]-dlon, lat2d[j,i]-dlat
                    h0 = h_itp.ev(lat0,lon0)
                    h1 = h_itp.ev(lat1,lon1)
                    if h0>h1: thetam += np.pi
                    theta_max[m,n,j,i] = thetam


 
# ------------ 保存输出到netcdf文件 ------------ 
if rank==0:
    print('####################################################################################')
    print('############# 保存输出到netcdf文件 #########################################')
    print('####################################################################################')
clock_diff = datetime.now() - clock
hour,sec = divmod(clock_diff.seconds,3600)
hour     = hour + clock_diff.days*24
minu,sec = divmod(sec,60)
print(' ===> 处理器 %.3i, 已用时间 : %.2i h %.2i min %.2i sec, 保存到netcdf文件 '\
      %(rank,hour,minu,sec))
nc = Dataset(file_output,'w')
nc.createDimension('nmonths',12) 
nc.createDimension('nmodes',nmodes) 
nc.createDimension('nlon',nlon) 
nc.createDimension('nlat',nlat)
if debug:
    nc.createDimension('npts_max',npts_max)
nc.createVariable('lon','f',('nlat','nlon')) 
nc.createVariable('lat','f',('nlat','nlon')) 
nc.createVariable('h','f',('nlat','nlon')) 
nc.createVariable('N2b','f',('nmonths','nlat','nlon')) 
nc.createVariable('k1','f',('nmonths','nlat','nlon')) 
nc.createVariable('npts','i',('nlat','nlon')) 
nc.createVariable('gamma_sup_avg','f',('nmonths','nlat','nlon')) 
nc.createVariable('gamma_sup_frac','f',('nmonths','nlat','nlon')) 
nc.createVariable('Eft','f',('nmonths','nlat','nlon')) 
nc.createVariable('Eft_sc','f',('nmonths','nlat','nlon')) 
nc.createVariable('Efn','f',('nmonths','nmodes','nlat','nlon')) 
nc.createVariable('Efn_sc','f',('nmonths','nmodes','nlat','nlon')) 
nc.createVariable('theta_max','f',('nmonths','nmodes','nlat','nlon')) 
if debug:
    nc.createVariable('Efa','f',('nmonths','nlat','nlon','npts_max')) 
    nc.createVariable('kh','f',('nlat','nlon','npts_max')) 
nc.variables['lon'][:]  = lon2d
nc.variables['lat'][:]  = lat2d
nc.variables['h'][:]    = h_grid 
nc.variables['N2b'][:]  = N2b 
nc.variables['k1'][:]   = k1
nc.variables['npts'][:] = npts
nc.variables['Eft'][:]  = Eft
nc.variables['Eft_sc'][:]  = Eft_sc
nc.variables['Efn'][:]  = Efn
nc.variables['Efn_sc'][:]  = Efn_sc
nc.variables['gamma_sup_avg'][:]  = gamma_sup_avg
nc.variables['gamma_sup_frac'][:]  = gamma_sup_frac
nc.variables['theta_max'][:]  = theta_max
if debug:
    nc.variables['kh'][:]   = kk 
    nc.variables['Efa'][:]  = Efa

nc.close()
``` 
