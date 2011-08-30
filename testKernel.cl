// MMT is calculating C = A*A',
// where A is a width x height type matrix
// TODO: localisation

// __kernel 
// void MMTold(__global float* a, 
// 	 __global float* c,
// 	 const unsigned int width,
// 	 const unsigned int height
// 	)
// {
//     uint i = get_global_id(0);
//     uint j = get_global_id(1);
// 
//     float sum = 0;
// 
//     for(int k=0; k<width; k++){
// 
//       sum+=a[j*width+k]*a[i*width+k];
//     }
// 
//     c[j*width+i]=sum;
// 
// //     CODE for M.Transpose
// //     c[i*height+j]=a[j*width+i];
// }

// Blocked version of MMT
__kernel 
void MMT(__global float* a, 
	 __global float* c,
	 const unsigned int width,
	 const unsigned int height,
	 __local  float* sdata
	)
{
    uint li = get_local_id(0);
    uint lj = get_local_id(1);

    __private uint lsi = get_local_size(0);
    __private uint lsj = get_local_size(1);

    uint col = get_group_id(0)*lsi + li;
    uint row = get_group_id(1)*lsj + lj;

    float sum = 0;

    for(int k=0; k<width; k++){
	float element1 = a[row*width+k];
	float element2 = a[col*width+k];
	sum += element1*element2;
    }
    c[row*width+col]=sum;
}

// PNorm2 is calculating c = ||a||^2,
// where a is a vector
// NOTE: size-padding needed!
__kernel 
void PNorm2G(__global float* a, 
	    __global float* c,
	    __local  float* sdata
	   )
{
    uint li = get_local_id(0);
    uint bi = get_group_id(0);
    __private uint ls = get_local_size(0);
    uint j  = bi*ls*2+li;

    sdata[li]=a[j]*a[j]+a[j+ls]*a[j+ls];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ls >= 512) {if (li < 256) { sdata[li] += sdata[li + 256];} barrier(CLK_LOCAL_MEM_FENCE); }
    if (ls >= 256) {if (li < 128) { sdata[li] += sdata[li + 128];} barrier(CLK_LOCAL_MEM_FENCE); }
    if (ls >= 128) {if (li <  64) { sdata[li] += sdata[li +  64];} barrier(CLK_LOCAL_MEM_FENCE); }

    if (li < 32) {
      if (ls >= 64) { sdata[li] += sdata[li +  32]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >= 32) { sdata[li] += sdata[li +  16]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >= 16) { sdata[li] += sdata[li +   8]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  8) { sdata[li] += sdata[li +   4]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  4) { sdata[li] += sdata[li +   2]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  2) { sdata[li] += sdata[li +   1]; barrier(CLK_LOCAL_MEM_FENCE); }
    }

    if ( li==0 ) {c[0] = sqrt(sdata[0]);}
}

__kernel 
void PNorm2L(__global float* a, 
	    __local  float* c,
	    __local  float* sdata
	   )
{
//     __private uint gi = get_global_id(0);
    uint li = get_local_id(0);
    uint bi = get_group_id(0);
    __private uint ls = get_local_size(0);
    uint j  = bi*ls*2+li;

    sdata[li]=a[j]*a[j]+a[j+ls]*a[j+ls];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ls >= 512) {if (li < 256) { sdata[li] += sdata[li + 256];} barrier(CLK_LOCAL_MEM_FENCE); }
    if (ls >= 256) {if (li < 128) { sdata[li] += sdata[li + 128];} barrier(CLK_LOCAL_MEM_FENCE); }
    if (ls >= 128) {if (li <  64) { sdata[li] += sdata[li +  64];} barrier(CLK_LOCAL_MEM_FENCE); }

    if (li < 32) {
      if (ls >= 64) { sdata[li] += sdata[li +  32]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >= 32) { sdata[li] += sdata[li +  16]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >= 16) { sdata[li] += sdata[li +   8]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  8) { sdata[li] += sdata[li +   4]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  4) { sdata[li] += sdata[li +   2]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  2) { sdata[li] += sdata[li +   1]; barrier(CLK_LOCAL_MEM_FENCE); }
    }

    if ( li==0 ) {c[0] = sqrt(sdata[0]);}
}

// PNorm2v2 is calculating c = a/||a||,
// where a is a vector
// TODO: size-padding

__kernel 
void PNorm2v2(__global float* a, 
	      __global float* c,
// 	      const unsigned int ei,
// 	      const float sigma,
	      __local  float* sdata,
	      __local  float* i
	     )
{
    PNorm2L(a, i ,sdata);
    
    uint gi = get_global_id(0);
    uint li = get_local_id(0);
    uint bi = get_group_id(0);
    uint ls = get_local_size(0);
    uint j  = bi*ls*2+li;

    c[gi]   = a[gi]/i[0];
    c[gi+ls]= a[gi+ls]/i[0];
}

// __kernel 
// void matrixTranspose(__global float * output,
//                      __global float * input,
//                      __local  float * block,
//                      const    uint    width,
//                      const    uint    height,
//                      const    uint blockSize
//                        )

//     uint localIdx = get_local_id(0);
//     uint localIdy = get_local_id(1);
//         
//     /* copy from input to local memory */
//     block[localIdy*blockSize + localIdx] = input[globalIdy*width + globalIdx];
// 
//     /* wait until the whole block is filled */
//     barrier(CLK_LOCAL_MEM_FENCE);
// 
//     uint groupIdx = get_group_id(0);
//     uint groupIdy = get_group_id(1);
// 
//     /* calculate the corresponding target location for transpose  by inverting x and y values*/
//     uint targetGlobalIdx = groupIdy*blockSize + localIdy;
//     uint targetGlobalIdy = groupIdx*blockSize + localIdx;
// 
//     /* calculate the corresponding raster indices of source and target */
//     uint targetIndex  = targetGlobalIdy*height     + targetGlobalIdx;
//     uint sourceIndex  = localIdy       * blockSize + localIdx;
//         
//      output[targetIndex] = block[sourceIndex];
