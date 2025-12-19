#include "stdio.h"
#include "global.hpp"
#include "Detector.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef __CUDACC__
#include "pybind11_cuda_array_interface.hpp"
#endif

namespace py = pybind11;


CUDA_DEVICE
int Orbits::get_window(double t)
{
    int out = int(t / dt);
    if ((out < 0) || (out >= N))
        return -1;
    else
        return out;
}

CUDA_DEVICE
int Orbits::get_link_ind(int link)
{
    if (link == 12)
        return 0;
    else if (link == 23)
        return 1;
    else if (link == 31)
        return 2;
    else if (link == 13)
        return 3;
    else if (link == 32)
        return 4;
    else if (link == 21)
        return 5;
    else
#ifdef __CUDACC__
        // printf("BAD link ind. Must be 12, 23, 31, 13, 32, 21.");
#else
        throw std::invalid_argument("Bad link ind. Must be 12, 23, 31, 13, 32, 21.");
#endif // __CUDACC__
    return -1;
}

CUDA_DEVICE
int Orbits::get_sc_ind(int sc)
{
    if (sc == 1)
        return 0;
    else if (sc == 2)
        return 1;
    else if (sc == 3)
        return 2;
    else
    {
#ifdef __CUDACC__
        // printf("BAD sc ind. Must be 1,2,3. %d\n", sc);
#else
        std::ostringstream oss;
        oss << "Bad sc ind. Must be 1,2,3. Input sc is " << sc << " " << std::endl;
        std::string var = oss.str();
        throw std::invalid_argument(var);
#endif // __CUDACC__
    }
    return 0;
}

CUDA_DEVICE
double Orbits::interpolate(double t, double *in_arr, int window, int major_ndim, int major_ind, int ndim, int pos)
{
    double up = in_arr[((window + 1) * major_ndim + major_ind) * ndim + pos]; // down_ind * ndim + pos];
    double down = in_arr[(window * major_ndim + major_ind) * ndim + pos];

    // m *(x - x0) + y0
    double fin = ((up - down) / dt) * (t - (dt * window)) + down;
    // if ((ndim == 1))
    //     printf("%d %e %e %e %e \n", window, fin, down, up, (t - (dt * window)));

    return fin;
}

CUDA_DEVICE
void Orbits::get_normal_unit_vec_ptr(Vec *vec, double t, int link)
{
    Vec _tmp = get_normal_unit_vec(t, link);
    vec->x = _tmp.x;
    vec->y = _tmp.y;
    vec->z = _tmp.z;
}

CUDA_DEVICE
Vec Orbits::get_normal_unit_vec(double t, int link)
{
    int window = get_window(t);
    if (window == -1)
    {
        // out of bounds
        return Vec(0.0, 0.0, 0.0);
    }

    int link_ind = get_link_ind(link);

    int up_ind = (window + 1) * nlinks + link_ind;
    int down_ind = window * nlinks + link_ind;

    // x (pos = 0) ndim = 3
    double x_out = interpolate(t, n_arr, window, nlinks, link_ind, 3, 0);
    // y (pos = 1)
    double y_out = interpolate(t, n_arr, window, nlinks, link_ind, 3, 1);
    // z (pos = 2)
    double z_out = interpolate(t, n_arr, window, nlinks, link_ind, 3, 2);

    return Vec(x_out, y_out, z_out);
}

CUDA_DEVICE
double Orbits::get_light_travel_time(double t, int link)
{
    int window = get_window(t);
    if (window == -1)
    {
        // out of bounds
        return 0.0;
    }

    int link_ind = get_link_ind(link);
    int up_ind = (window + 1) * (nlinks + link_ind);
    int down_ind = window * (nlinks + link_ind);

    // x (pos = 0), ndim = 1
    double ltt_out = interpolate(t, ltt_arr, window, nlinks, link_ind, 1, 0);

    return ltt_out;
}

CUDA_DEVICE
Vec Orbits::get_pos(double t, int sc)
{
    int window = get_window(t);
    if (window == -1)
    {
        // out of bounds
        return Vec(0.0, 0.0, 0.0);
    }

    int sc_ind = get_sc_ind(sc);

    // x (pos = 0), ndim = 3
    double x_out = interpolate(t, x_arr, window, nspacecraft, sc_ind, 3, 0);
    // y (pos = 1), ndim = 3
    double y_out = interpolate(t, x_arr, window, nspacecraft, sc_ind, 3, 1);
    // z (pos = 2), ndim = 3
    double z_out = interpolate(t, x_arr, window, nspacecraft, sc_ind, 3, 2);
    return Vec(x_out, y_out, z_out);
}

CUDA_DEVICE
void Orbits::get_pos_ptr(Vec *vec, double t, int sc)
{
    Vec _tmp = get_pos(t, sc);
    vec->x = _tmp.x;
    vec->y = _tmp.y;
    vec->z = _tmp.z;
}

#define NUM_THREADS 64


CUDA_KERNEL
void get_light_travel_time_kernel(double *ltt, double *t, int *link, int num, Orbits &orbits)
{
    int start, end, increment;
#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;
#else  // __CUDACC__
    start = 0;
    end = num;
    increment = 1;
#endif // __CUDACC__

    for (int i = start; i < end; i += increment)
    {
        ltt[i] = orbits.get_light_travel_time(t[i], link[i]);
    }
}


void Orbits::get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num)
{
    get_light_travel_time_arr(
        return_pointer_and_check_length(ltt, "ltt", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(link, "sc", num, 1),
        num
    );
}

void Orbits::get_light_travel_time_arr(double *ltt, double *t, int *link, int num)
{
#ifdef __CUDACC__
    int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

    // copy self to GPU
    Orbits *orbits_gpu;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, this, sizeof(Orbits), cudaMemcpyHostToDevice));

    get_light_travel_time_kernel<<<num_blocks, NUM_THREADS>>>(ltt, t, link, num, *orbits_gpu);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));

#else // __CUDACC__

    get_light_travel_time_kernel(ltt, t, link, num, *this);

#endif // __CUDACC__
}


CUDA_KERNEL
void get_pos_kernel(double *pos_x, double *pos_y, double *pos_z, double *t, int *sc, int num, Orbits &orbits)
{
    int start, end, increment;
#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;
#else  // __CUDACC__
    start = 0;
    end = num;
    increment = 1;
#endif // __CUDACC__
    Vec _tmp(0.0, 0.0, 0.0);

    for (int i = start; i < end; i += increment)
    {
        _tmp = orbits.get_pos(t[i], sc[i]);
        pos_x[i] = _tmp.x;
        pos_y[i] = _tmp.y;
        pos_z[i] = _tmp.z;
    }
}


void Orbits::get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num)
{
    get_pos_arr(
        return_pointer_and_check_length(pos_x, "pos_x", num, 1),
        return_pointer_and_check_length(pos_y, "pos_y", num, 1),
        return_pointer_and_check_length(pos_z, "pos_z", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(sc, "sc", num, 1),
        num
    );
}

void Orbits::get_pos_arr(double *pos_x, double *pos_y, double *pos_z, double *t, int *sc, int num)
{
#ifdef __CUDACC__
    int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

    // copy self to GPU
    Orbits *orbits_gpu;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, this, sizeof(Orbits), cudaMemcpyHostToDevice));

    get_pos_kernel<<<num_blocks, NUM_THREADS>>>(pos_x, pos_y, pos_z, t, sc, num, *orbits_gpu);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));

#else // __CUDACC__

    get_pos_kernel(pos_x, pos_y, pos_z, t, sc, num, *this);

#endif // __CUDACC__
}


CUDA_KERNEL
void get_normal_unit_vec_kernel(double *normal_unit_vec_x, double *normal_unit_vec_y, double *normal_unit_vec_z, double *t, int *link, int num, Orbits &orbits)
{
    int start, end, increment;
#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;
#else  // __CUDACC__
    start = 0;
    end = num;
    increment = 1;
#endif // __CUDACC__
    Vec _tmp(0.0, 0.0, 0.0);

    for (int i = start; i < end; i += increment)
    {
        _tmp = orbits.get_normal_unit_vec(t[i], link[i]);
        normal_unit_vec_x[i] = _tmp.x;
        normal_unit_vec_y[i] = _tmp.y;
        normal_unit_vec_z[i] = _tmp.z;
    }
}

void Orbits::get_normal_unit_vec_arr(double *normal_unit_vec_x, double *normal_unit_vec_y, double *normal_unit_vec_z, double *t, int *link, int num)
{
#ifdef __CUDACC__
    int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

    // copy self to GPU
    Orbits *orbits_gpu;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, this, sizeof(Orbits), cudaMemcpyHostToDevice));

    get_normal_unit_vec_kernel<<<num_blocks, NUM_THREADS>>>(normal_unit_vec_x, normal_unit_vec_y, normal_unit_vec_z, t, link, num, *orbits_gpu);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));

#else // __CUDACC__

    get_normal_unit_vec_kernel(normal_unit_vec_x, normal_unit_vec_y, normal_unit_vec_z, t, link, num, *this);

#endif // __CUDACC__
}

void Orbits::get_normal_unit_vec_wrap(array_type<double>normal_unit_vec_x, array_type<double>normal_unit_vec_y, array_type<double>normal_unit_vec_z, array_type<double>t, array_type<int>link, int num)
{
    
// #ifdef __CUDACC__
    get_normal_unit_vec_arr(
        return_pointer_and_check_length(normal_unit_vec_x, "n_arr_x", num, 1),
        return_pointer_and_check_length(normal_unit_vec_y, "n_arr_y", num, 1),
        return_pointer_and_check_length(normal_unit_vec_z, "n_arr_z", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(link, "link", num, 1),
        num
    );
}

// PYBIND11_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
PYBIND11_MODULE(pycppdetector, m) {
    m.doc() = "Orbits/Detector C++ plug-in"; // Optional module docstring

    py::class_<Orbits>(m, "Orbits")
    // Bind the constructor
    .def(py::init<double, int, array_type<double>, array_type<double>, array_type<double>, array_type<int>, array_type<int>, array_type<int>, double>(), 
         py::arg("dt"), py::arg("N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))
    // Bind member functions
    .def("get_light_travel_time_wrap", &Orbits::get_light_travel_time_wrap, "Get the light travel time.")
    .def("get_pos_wrap", &Orbits::get_pos_wrap, "Get spacecraft position.")
    .def("get_normal_unit_vec_wrap", &Orbits::get_normal_unit_vec_wrap, "Get link normal vector.")
    // You can also expose public data members directly using def_readwrite
    ;
}

