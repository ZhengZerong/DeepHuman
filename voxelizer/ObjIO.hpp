#ifndef __OBJIO_HPP__
#define __OBJIO_HPP__

#include <iostream>
#include <fstream>
#include <vector_types.h>
#include <math.h>
#include <string>
#include <vector>

float readFloat(std::ifstream &_ifs)
{
    char buf[32] = { 0 };
    int pt = 0;

    while (!_ifs.eof())
    {
        char c = 0;
        _ifs.read(&c, sizeof(char));
        if (c == ' ' || c == '\n') break;
        buf[pt] = c;
        pt++;
    };

    if (pt == 0) return 0.f;
    return std::stof(buf);
}

float readInt(std::ifstream &_ifs)
{
    char buf[32] = { 0 };
    int pt = 0;

    while (!_ifs.eof())
    {
        char c = 0;
        _ifs.read(&c, sizeof(char));
        if (c == ' ' || c == '\n') break;
        buf[pt] = c;
        pt++;
    };

    if (pt == 0) return 0.f;
    return std::stoi(buf);
}

void loadFromObjBinary(std::vector<float3> &_vs, std::vector<int3> &_fs, const std::string &_filename)
{
    _vs.clear();
    _fs.clear();
    std::ifstream ifile(_filename, std::ifstream::binary);
    if (!ifile.is_open())
    {
        std::cout << "Failed to open " << _filename << std::endl;
        return;
    }

    char vf, sp;
    while (!ifile.eof())
    {
        ifile.read(&vf, sizeof(char));
        if (ifile.eof()) break;
        if (vf == 'v')
        {
            float3 v;
            ifile.read(&sp, sizeof(char)); //space
            v.x = readFloat(ifile);
            v.y = readFloat(ifile);
            v.z = readFloat(ifile);
            _vs.push_back(v);
        }
        else if (vf == 'f')
        {
            int3 f;
            ifile.read(&sp, sizeof(char)); //space
            f.x = readInt(ifile)-1;
            f.y = readInt(ifile)-1;
            f.z = readInt(ifile)-1;
            _fs.push_back(f);
        }
    }

    std::cout << "Loaded " << _vs.size() << " vertices and " << _fs.size() << " triangles. \n";
}

void saveVolume(const std::vector<int> &_volume, int3 _vol_res, float3 _vol_min_corner, float3 _vol_max_corner, const std::string &_filename)
{
    float3 step;
    step.x = (_vol_max_corner.x - _vol_min_corner.x) / _vol_res.x;
    step.y = (_vol_max_corner.y - _vol_min_corner.y) / _vol_res.y;
    step.z = (_vol_max_corner.z - _vol_min_corner.z) / _vol_res.z;
    float boxhalfsize[3] = { step.x / 2.f, step.y / 2.f, step.z / 2.f };

    std::ofstream ofile(_filename);
    for (int xx = 0; xx < _vol_res.x; xx++)
    {
        for (int yy = 0; yy < _vol_res.y; yy++)
        {
            for (int zz = 0; zz < _vol_res.z; zz++)
            {
                int id = zz*_vol_res.y*_vol_res.x + yy*_vol_res.x + xx;
                if (_volume[id] > 0)
                {
                    float boxcenter_x = xx*step.x + boxhalfsize[0] + _vol_min_corner.x;
                    float boxcenter_y = yy*step.y + boxhalfsize[1] + _vol_min_corner.y;
                    float boxcenter_z = zz*step.z + boxhalfsize[2] + _vol_min_corner.z;

                    ofile << "v " << boxcenter_x << " " << boxcenter_y << " " << boxcenter_z << "\n";
                }
            }
        }
    }

    ofile.close();
}

void saveVolumeNonzeroIdx(const std::vector<int> &_volume, int3 _vol_res, const std::string &_filename)
{
    std::ofstream ofile(_filename);
    for (int xx = 0; xx < _vol_res.x; xx++)
    {
        for (int yy = 0; yy < _vol_res.y; yy++)
        {
            for (int zz = 0; zz < _vol_res.z; zz++)
            {
                int id = zz*_vol_res.y*_vol_res.x + yy*_vol_res.x + xx;
                if (_volume[id] > 0)
                {
                    ofile << xx << " " << yy << " " << zz << "\n";
                }
            }
        }
    }
    ofile.close();
}

#endif