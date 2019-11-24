#include <iostream>
#include <fstream>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string>
#include <vector>

#include "ObjIO.hpp"
#include "safecall.hpp"

void setIntersectVoxels(float3 *_vertices_ptr, int3* _faces_ptr, int* _volume, int3 _vol_res, float3 _vol_min_corner, float3 _vol_max_corner, int _vertice_num, int _face_num);

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "*************************************************************************\n";
        std::cout << "*  Fast Voxelizer\n";
        std::cout << "*  A fast voxelization algorithm implemented in C++ with CUDA support. \n";
        std::cout << "* \n";
        std::cout << "*  Usage: ./voxelizer path/to/obj/file path/to/output/file \n";
        std::cout << "* \n";
        std::cout << "*  Reference: \n";
        std::cout << "*     Tomas Akenine-Moller. Fast 3D Triangle-Box Overlap Testing. \n";
        std::cout << "*     http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller\n";
        std::cout << "*     \code/tribox3.txt\n";
        std::cout << "*************************************************************************\n";
        return -1;
    }

     std::string fname(argv[1]);
     std::string fout_name(argv[2]);
     std::vector<float3> vs;
     std::vector<int3> fs;
     loadFromObjBinary(vs, fs, fname);
     std::cout << "Processing...";

     float3 vol_min_corner = make_float3(-0.3333333, -0.5, -0.3333333);
     float3 vol_max_corner = make_float3(0.3333333, 0.5, 0.3333333);
     int3 vol_res = make_int3(128, 192, 128);

     float3* vs_dev;
     int3* fs_dev;
     cudaSafeCall(cudaMalloc((void**)&vs_dev, sizeof(float3)*vs.size()));
     cudaSafeCall(cudaMalloc((void**)&fs_dev, sizeof(int3)*fs.size()));
     cudaSafeCall(cudaMemcpy((void*)vs_dev, (void*)vs.data(), sizeof(float3)*vs.size(), cudaMemcpyHostToDevice));
     cudaSafeCall(cudaMemcpy((void*)fs_dev, (void*)fs.data(), sizeof(int3)*fs.size(), cudaMemcpyHostToDevice));
     std::cout << ".....";

     int* volume_dev;
     cudaSafeCall(cudaMalloc((void**)&volume_dev, sizeof(int)*vol_res.x*vol_res.y*vol_res.z));
     cudaSafeCall(cudaMemset((void*)volume_dev, 0, sizeof(int)*vol_res.x*vol_res.y*vol_res.z));
     setIntersectVoxels(vs_dev, fs_dev, volume_dev, vol_res, vol_min_corner, vol_max_corner, vs.size(), fs.size());
     std::cout << ".....";

     std::vector<int> volume;
     volume.resize(vol_res.x*vol_res.y*vol_res.z, 0);
     cudaSafeCall(cudaMemcpy((void*)volume.data(), (void*)volume_dev, sizeof(int)*volume.size(), cudaMemcpyDeviceToHost));
     saveVolumeNonzeroIdx(volume, vol_res, fout_name);
     std::cout << ".....";
     std::cout << "done\n";

     return 0;
}