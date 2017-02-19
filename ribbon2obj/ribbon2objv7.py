#coding:utf-8
#圆柱
import numpy as np 
import scipy.interpolate
import sys
import argparse

#解析 .pdb.ribbon.txt文件，获取个Residue的(x,y,z)位置坐标
'''
lines = open(ribbon_path).readlines()
num = len(lines)
aid = range(0,num)	# 0,1,2,...
x = [0] * num
y = [0] * num
z = [0] * num
for line in lines:
	i,atype,xi,yi,zi,nxi,nyi,nzi,bxi,byi,bzi = map(float,line.strip().split(','))
	i = int(i)
	aid[i] = i
	x[i] = xi
	y[i] = yi
	z[i] = zi
'''
#解析 .pdb文件，使用 ATOM CA 获取个Residue的(x,y,z)位置坐标
def residue2xyz(residue_path):
	lines = open(residue_path).readlines()
	num = len(lines)
	aid = range(0,num)
	i = 0
	x = [0] * num
	y = [0] * num
	z = [0] * num
	for line in lines:
		xi,yi,zi = map(float,line.strip().split(','))
		i = int(i)
		aid[i] = i
		x[i] = xi
		y[i] = yi
		z[i] = zi
		i = i + 1
	return aid,x,y,z
def ribbon2xyzbn(ribbon_path,type='default'):	
	#aid,x,y,z,bvector,nvector = ribbon2xyzbn(ribbon_path)
	lines = open(ribbon_path).readlines()
	num = len(lines)
	aid = range(0,num)	# 0,1,2,...
	x = [0] * num
	y = [0] * num
	z = [0] * num
	bvector = [0] * num
	nvector = [0] * num
	for line in lines:
		i,atype,xi,yi,zi,nxi,nyi,nzi,bxi,byi,bzi = map(float,line.strip().split(','))
		i = int(i)
		aid[i] = i
		x[i] = xi
		y[i] = yi
		z[i] = zi
		bvector[i] = np.array([bxi,byi,bzi])
		nvector[i] = np.array([nxi,nyi,nzi])	
	
	if type=='cylinder':	
		return aid,x,y,z,nvector,bvector #交换一下
	else:
		return aid,x,y,z,bvector,nvector
#中轴线插值
#interpolate : (aid,x) => (aidnew,xnew)
def xyz2new(aid,x,y,z,interpolate_factor):
	num = len(aid)
	aid = np.array(aid)
	aidnew = np.linspace(aid[0],aid[-1],interpolate_factor*num)

	fx = scipy.interpolate.interp1d(aid,x,'cubic')
	xnew = fx(aidnew)

	fy = scipy.interpolate.interp1d(aid,y,'cubic')
	ynew = fy(aidnew)

	fz = scipy.interpolate.interp1d(aid,z,'cubic')
	znew = fz(aidnew)

	numnew = len(xnew)
	return aidnew,xnew,ynew,znew
#中轴线->vcvf
def xyzbn2vcvf(x,y,z,bvector,nvector,wb,wn,interpolate_factor,type='default',nj=4,ndim=3):
	num = len(x)
	numnew = num * interpolate_factor
	aid = range(0,num)
	aidnew = np.linspace(aid[0],aid[-1],numnew)	
	aidnew,xnew,ynew,znew = xyz2new(aid,x,y,z,interpolate_factor)
	vc_list_new = map(lambda i:np.array([xnew[i],ynew[i],znew[i]]),range(0,numnew))

	vc_list = [0] * num # center vertex
	vf_list = [0] * num # [[v0,v1,v2,v3], [v0,v1,v2,v3], [v0,v1,v2,v3], ...] 

	if type=='default':
		for i in aid:
			#vx [vx0,vx1,vx2,vx3]
			# wb, wn 
			# vc: center vertex point
			vc = np.array([x[i],y[i],z[i]])
			bv = bvector[i]
			nv = nvector[i]
			v0 = vc + wb * bv + wn * nv
			v1 = vc - wb * bv + wn * nv
			v2 = vc - wb * bv - wn * nv
			v3 = vc + wb * bv - wn * nv
			vc_list[i] = vc
			vf_list[i] = [v0,v1,v2,v3]
		_v = [[], [], [], []]
		_vnew = [[], [], [], []]	# len = 10 * num	
		for j in range(0,4):	# v0,v1,v2,v3
			_v[j] = [[],[],[]]
			_vnew[j] = [[],[],[]]	# _vnewjx,_vnewjy,_vnewjz
			for dim in range(0,3):
				_vj = np.array(map(lambda v:v[j],vf_list))
				_vjd = map(lambda _vj:_vj[dim], _vj)
				_v[j][dim] = _vjd
				f = scipy.interpolate.interp1d(aid, _vjd,'cubic')
				_vjdnew = f(aidnew)
				_vnew[j][dim] = _vjdnew
		vf_list_new = [0] * numnew
		for i in range(0,numnew):
			vf_list_new[i] = map(lambda j:np.array(map(lambda dim:_vnew[j][dim][i],range(0,3))),range(0,4))
	
	elif type=='cylinder':
		#nj = 12 # 参数，圆柱侧面切分的分数

		for i in aid:
			#vx [vx0,vx1,vx2,vx3]
			# wb, wn 
			# vc: center vertex point
			vc = np.array([x[i],y[i],z[i]])
			bv = bvector[i]
			nv = nvector[i]
			vc_list[i] = vc
			vf_list[i] = [0] * nj
			for j in range(0,nj):
				vf_list[i][j] = vc + bv * wb * np.cos(2 * np.pi / nj * j) + nv * wn * np.sin(2 * np.pi / nj * j)
		_v = [0] * nj
		_vnew = [0] * nj	# len = 10 * num	
		for j in range(0,nj):	# v0,v1,v2,v3
			_v[j] = [0] * ndim
			_vnew[j] = [0] * ndim	# _vnewjx,_vnewjy,_vnewjz
			for dim in range(0,ndim):
				_vj = np.array(map(lambda v:v[j],vf_list))
				_vjd = map(lambda _vj:_vj[dim], _vj)
				_v[j][dim] = _vjd
				f = scipy.interpolate.interp1d(aid, _vjd,'cubic')
				_vjdnew = f(aidnew)
				_vnew[j][dim] = _vjdnew
		vf_list_new = [0] * numnew

		for i in range(0,numnew):
			vf_list_new[i] = map(lambda j:np.array(map(lambda dim:_vnew[j][dim][i],range(0,ndim))),range(0,nj))
	
	
	return vc_list_new,vf_list_new
# 主、副法向量计算
def xyz2bn(xnew,ynew,znew,bvector_threshold):
	numnew = len(xnew)
	bvector_new = [0] * numnew
	nvector_new = [0] * numnew
	# 头、尾部的点的主、副法向量单独计算。其他点用前后各一个点计算。
	# 头部

	i = 0 
	tangent_1 = np.array([xnew[i+1]-xnew[i],ynew[i+1]-ynew[i],znew[i+1]-znew[i]])
	tangent_2 = np.array([xnew[i+2]-xnew[i+1],ynew[i+2]-ynew[i+1],znew[i+2]-znew[i+1]])
	bvector_0 = np.cross(tangent_1,tangent_2)	#np.cross: 叉乘
	bvector_0_norm = sum(bvector_0**2)**0.5	
	bvector_new[0] = bvector_0 / bvector_0_norm
	nvector_new[0] = np.cross(tangent_1,bvector_new[0])

	#腹部
	for i in range(1,numnew-1):
		tangent_0 = np.array([xnew[i]-xnew[i-1],ynew[i]-ynew[i-1],znew[i]-znew[i-1]])
		tangent_1 = np.array([xnew[i+1]-xnew[i],ynew[i+1]-ynew[i],znew[i+1]-znew[i]])
		#if np.dot(tangent_0,tangent_1)/np.sqrt(np.dot(tangent_0,tangent_0)*np.dot(tangent_1,tangent_1)) > 
		#鲁棒性考虑，当两个tangent的夹角足够大时才用叉积计算，否则保持旧值。
		bvector_i = np.cross(tangent_0,tangent_1)	#np.cross: 叉乘
		bvector_i_norm = sum(bvector_i**2)**0.5	
		if bvector_i_norm / np.sqrt(np.dot(tangent_0,tangent_0)*np.dot(tangent_1,tangent_1)) > bvector_threshold:
			bvector_i = bvector_i / bvector_i_norm	#单位化
		else:
			bvector_i = bvector_new[i-1]
		nvector_i = np.cross(tangent_0+tangent_1,bvector_i)
		nvector_i = nvector_i / sum(nvector_i**2)**0.5	#单位化

		bvector_new[i] = bvector_i
		nvector_new[i] = nvector_i

	# 尾部
	bvector_new[numnew-1] = bvector_new[numnew-2]
	i = numnew-1
	tangent_0 = np.array([xnew[i]-xnew[i-1],ynew[i]-ynew[i-1],znew[i]-znew[i-1]])
	nvector_new[numnew-1] = np.cross(tangent_0,bvector_new[0])

	return bvector_new,nvector_new


# vertex
#
# 3 2	7 6			 
# 0 1	4 5    b<-- | 
#					V n
def newbn2vcvf(xnew,ynew,znew,bvector_new,nvector_new,wb,wn):
	numnew = len(xnew)
	vc_list = [0] * numnew # center vertex
	vf_list = [0] * numnew # [[v0,v1,v2,v3], [v0,v1,v2,v3], [v0,v1,v2,v3], ...] 
	for i in range(0,numnew):
		#vx [vx0,vx1,vx2,vx3]
		# wb, wn 
		# vc: center vertex point
		vc = np.array([xnew[i],ynew[i],znew[i]])
		bvector_i = bvector_new[i]
		nvector_i = nvector_new[i]
		v0 = vc + wb * bvector_i + wn * nvector_i
		v1 = vc - wb * bvector_i + wn * nvector_i
		v2 = vc - wb * bvector_i - wn * nvector_i
		v3 = vc + wb * bvector_i - wn * nvector_i
		vc_list[i] = vc
		vf_list[i] = [v0,v1,v2,v3]
	return vc_list,vf_list
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
vf_flatlist = []
for vf in vf_list:
	for v in vf:
		vf_flatlist.append(v)
_x = np.array(map(lambda v:v[0],vf_flatlist))
_y = np.array(map(lambda v:v[1],vf_flatlist))
_z = np.array(map(lambda v:v[2],vf_flatlist))
ax.plot(_x,_y,_z)
plt.show()
'''

#obj
# obj模型文件里的顶点v部分
# 按以下螺旋顺序
# 3 2	7 6			 
# 0 1	4 5    b<-- | 
#					V n
def vf2strv(vf_list):
	numnew = len(vf_list)
	str_v = ''
	for i in range(0,numnew):
		# vf = [v0,v1,v2,v3]
		vf = vf_list[i]
		for vi in vf:
			str_v = str_v + 'v %g %g %g\n' %(vi[0],vi[1],vi[2])
	return str_v

# obj模型文件里的顶点法向量vn部分，等同于主、副法向量。
# [ bvector,nvector, - bvector, - nvector, ]起点，终点
def bn2newbn(vf_list_new):
	numnew = len(vf_list_new)
	bvector_new = [0] * numnew
	nvector_new = [0] * numnew
	for i in range(0,numnew):
		bvector_new[i] = vf_list_new[i][0] - vf_list_new[i][1]
		nvector_new[i] = vf_list_new[i][3] - vf_list_new[i][0]
	return bvector_new,nvector_new
def bn2strvn(bvector_new,nvector_new, type='default'):
	numnew = len(bvector_new)
	str_vn = ''
	vn_list = []
	if type == 'default':
		for i in range(0,numnew):
			bv = bvector_new[i]
			nv = nvector_new[i]
			str_vn = str_vn + 'vn %g %g %g\n' %(bv[0],bv[1],bv[2])
			vn_list.append(bv)
			str_vn = str_vn + 'vn %g %g %g\n' %(nv[0],nv[1],nv[2])
			vn_list.append(nv)
			str_vn = str_vn + 'vn %g %g %g\n' %(-bv[0],-bv[1],-bv[2])
			vn_list.append(-bv)
			str_vn = str_vn + 'vn %g %g %g\n' %(-nv[0],-nv[1],-nv[2])
			vn_list.append(-nv)
		#底面和顶面的vn顶点法向量
		tangent_0 = np.cross(bvector_new[0],nvector_new[0])
		tangent_1 = np.cross(bvector_new[-1],nvector_new[-1])
		str_vn = str_vn + 'vn %g %g %g\n' %(tangent_1[0],tangent_1[1],tangent_1[2])
		vn_list.append(tangent_1)
		str_vn = str_vn + 'vn %g %g %g\n' %(-tangent_0[0],-tangent_0[1],-tangent_0[2])
		vn_list.append(-tangent_0)
	return str_vn, vn_list

#由vf_list得到str_vn
# 圆柱
def vflist2strvn(vc_list,vf_list):
	numnew = len(vf_list)
	vn_list = [0] * (numnew + 1)
	for i in range(0,numnew):
		vn_list[i] = [v - vc_list[i] for v in vf_list[i]]
	vn_list[numnew]= [vc_list[0] - vc_list[1], vc_list[-1] - vc_list[-2]]

	str_vn = ''
	for vni in vn_list:
	#for i in range(0,numnew):
		# vf = [v0,v1,v2,v3]
		#vni = vn_list[i]
		for vi in vni:
			str_vn = str_vn + 'vn %g %g %g\n' %(vi[0],vi[1],vi[2])
	
	return str_vn,vn_list

def getRibbonStrVN(vc_list,vf_list,bvector_new,nvector_new,type='cylinder'):
	if type == 'default':
		str_vn,vn_list = bn2strvn(bvector_new,nvector_new)
	elif type == 'cylinder':
		str_vn,vn_list = vflist2strvn(vc_list,vf_list)
	return str_vn,vn_list


# obj模型文件里的面f部分
# f v/vt/vn v/vt/vn v/vt/vn 

def getRibbonStrf(numnew, withVN=True, type='default', nj=4, ndim=3):
	str_f = ''
	str_f = str_f + ('g (null)\n')
	str_f = str_f + ('s 0\n')
	# 3 2	4 3			 
	# 0 1	1 2    b<-- | 
	#					V n

	

	# attention: right hand 
	if type == 'default':
		f_id_body=[
	# v,vn, v,vn, v,vn
			[1,4,6,8,2,4],
			[1,4,5,8,6,8],
			[2,3,6,7,7,7],
			[2,3,7,7,3,3],
			[3,2,8,6,4,2],
			[3,2,7,6,8,6],
			[4,1,5,5,1,1],
			[4,1,8,5,5,5]
		]
		if withVN is True:
			str_f = str_f + ('f 1//%d 2//%d 4//%d\n' %(numnew*2+1,numnew*2+1,numnew*2+1))
			str_f = str_f + ('f 2//%d 3//%d 4//%d\n' %(numnew*2+1,numnew*2+1,numnew*2+1))
			for i in range(0,numnew-1):
				vid_start = i*4
				for t in f_id_body: 
					str_f = str_f + ('f %d//%d %d//%d %d//%d\n' %(vid_start+t[0],vid_start+t[1],vid_start+t[2],vid_start+t[3],vid_start+t[4],vid_start+t[5]))
				
			vid_start = (numnew-1)*4
			str_f = str_f + ('f %d//%d %d//%d %d//%d\n' %(vid_start+2,numnew*2+2,vid_start+1,numnew*2+2,vid_start+4,numnew*2+2))
			str_f = str_f + ('f %d//%d %d//%d %d//%d\n' %(vid_start+3,numnew*2+2,vid_start+2,numnew*2+2,vid_start+4,numnew*2+2))
		else:
			str_f = str_f + ('f 1 2 4\n')
			str_f = str_f + ('f 2 3 4\n')
			for i in range(0,numnew-1):
				vid_start = i*4
				for t in f_id_body: 
					str_f = str_f + ('f %d %d %d\n' %(vid_start+t[0],vid_start+t[2],vid_start+t[4]))
				
			vid_start = (numnew-1)*4
			str_f = str_f + ('f %d %d %d\n' %(vid_start+2,vid_start+1,vid_start+4))
			str_f = str_f + ('f %d %d %d\n' %(vid_start+3,vid_start+2,vid_start+4))
	elif type == 'cylinder':
		def cylinder_f_yield(numnew, nj):
			#底面：正nj边形的三角剖分
				# 1 + (0, j, j+1),如(3,2,1),(4,3,1),(5,4,1),...(nj,nj-1,1)
			i = 0
			for j in range(2,nj):
				yield [j+1, nj*numnew+1, j, nj*numnew+1, 1, nj*numnew+1]
			#顶面
			i = numnew-1
			for j in range(2,nj):
				yield [j +i*nj, nj*numnew+2, j+1 +i*nj, nj*numnew+2, 1 +i*nj, nj*numnew+2]
			#侧面
			for i in xrange(0,numnew-1):
				for j in range(0,nj-1):
					yield [1 + i*nj + x for x in [j, j, j+1+nj, j+1+nj, j+nj, j+nj]]
					yield [1 + i*nj + x for x in [j, j, j+1, j+1, j+1+nj, j+1+nj]]
				j = nj - 1
				yield [1 + i*nj + x for x in [j, j, nj, nj, j+nj, j+nj]]
				yield [1 + i*nj + x for x in [j, j, 0, 0, nj, nj]]

		f_id_body = cylinder_f_yield(numnew, nj) 
		#list(f_id_body)	
		if withVN is True:
			for f in f_id_body:
				#str_f = str_f + 'f ' + ' '.join(map(str,f))+'\n'
				str_f = str_f + 'f %d//%d %d//%d %d//%d\n' %tuple(f)
		else:
			for f in f_id_body:
				#str_f = str_f + 'f ' + ' '.join(map(str,f[::2]))+'\n'
				str_f = str_f + 'f %d %d %d\n' %tuple(f[::2])
	return str_f	

def str2objfile(str_v,str_vn,str_f,obj3d_path):
	f_obj = open(obj3d_path,'w')
	f_obj.writelines('mtllib ./Box.mtl\n')
	#f_obj.writelines('# vertex_num = %d\n' %(numnew*4))
	f_obj.writelines(str_v)
	f_obj.writelines(str_vn)
	f_obj.writelines(str_f)
	f_obj.close()



#if __name__ == '__main__':
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--ribbon_path',help='input file test.pdb.ribbon.txt',default='test.pdb.ribbon.txt')
	parser.add_argument('-o','--obj3d_path',help='output file test.pdb.ribbon.txt.obj',default='test.pdb.ribbon.txt.obj')
	parser.add_argument('-f','--interpolate_factor',help='the interpolate_factor',type=int,default=10)
	parser.add_argument('-j','--nj',help='the edge number of cross section',type=int,default=20)
	parser.add_argument('-b','--wb',help='the weight of binormal vector',type=float)
	parser.add_argument('-n','--wn',help='the weight of normal vector',type=float)
	parser.add_argument('-t','--type',help='the type of the shape',choices=['default','cylinder'],default='default')
	parser.add_argument('--DEBUG',action="store_true")
	
	args = parser.parse_args()

	#默认参数值
	ribbon_path = args.ribbon_path # 'test.pdb.ribbon.txt'
	obj3d_path = args.obj3d_path # 'test.pdb.ribbon.txt.obj'

	if args.type == 'default':
		wb = 1		#条带宽度参数，与副法向量对应
		wn = 0.3	#条带厚度参数，与主法向量对应
	else:
		wb = 0.5
		wn = 0.5
	if args.wb:
		wb = args.wb
	if args.wn:
		wn = args.wn

	interpolate_factor = args.interpolate_factor # 10	#插值稠密倍数
	#bvector_threshold = 0.1	#副法向量叉积阈值
	nj = args.nj # 20
	type = args.type # 'default'
	DEBUG = args.DEBUG

	#关键节点处的主副法向量
	aid,x,y,z,bvector,nvector = ribbon2xyzbn(ribbon_path,type=type)

	#中轴线->矩形条带
	vc_list,vf_list = xyzbn2vcvf(x,y,z,bvector,nvector,wb,wn,interpolate_factor,type = type,nj=nj)
	
	#顶点坐标
	str_v = vf2strv(vf_list)

	#法向量
	bvector_new,nvector_new = bn2newbn(vf_list)

	#顶点法向量
	str_vn,vn_list = getRibbonStrVN(vc_list,vf_list,bvector_new,nvector_new,type=type)

	#面
	numnew = len(vf_list)
	str_f = getRibbonStrf(numnew,withVN=True, type=type, nj=nj, ndim=3)
	#num = len(aid)
	#str_f = getRibbonStrf(num,True)

	#写文件
	str2objfile(str_v,str_vn,str_f,obj3d_path)

	if  DEBUG:
		f_xyz = open('xyz.csv','w')
		f_bvector = open('bvector.csv','w')
		f_nvector = open('nvector.csv','w')
		for i in range(len(x)):
			f_xyz.writelines('%g,%g,%g\n' %(x[i],y[i],z[i]))
			f_bvector.writelines('%g,%g,%g\n' %(bvector_new[i][0],bvector_new[i][1],bvector_new[i][2]))
			f_nvector.writelines('%g,%g,%g\n' %(nvector_new[i][0],nvector_new[i][1],nvector_new[i][2]))
		f_xyz.close()
		f_bvector.close()
		f_nvector.close()

		f_xyznew = open('xyznew.csv','w')
		f_v = open('v.csv','w')
		f_vn = open('vn.csv','w')
		for i in range(0,numnew):
			f_xyznew.writelines('%g,%g,%g\n' %(vc_list[i][0],vc_list[i][1],vc_list[i][2]))
			vf = vf_list[i]
			for vi in vf:
				f_v.writelines('%g,%g,%g\n' %(vi[0],vi[1],vi[2]))
		f_xyznew.close()
		f_v.close()
		f_vn.close()

if __name__ == '__main__':
	main()