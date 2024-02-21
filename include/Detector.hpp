#ifndef __DETECTOR_HPP__
#define __DETECTOR_HPP__

class Vec
{
public:
    double x;
    double y;
    double z;

    Vec(double x_, double y_, double z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }
};

class Orbits
{
public:
    double dt;
    int N;
    double *n_arr;
    double *ltt_arr;
    double *x_arr;
    int nlinks;
    int nspacecraft;

    Orbits(double dt_, int N_, double *n_arr_, double *ltt_arr_, double *x_arr_)
    {
        dt = dt_;
        N = N_;
        n_arr = n_arr_;
        ltt_arr = ltt_arr_;
        x_arr = x_arr_;
        nlinks = 6;
        nspacecraft = 3;
    };

    int get_window(double t);
    Vec get_normal_unit_vec(double t, int link);
    double interpolate(double t, double *in_arr, int down_ind, int up_ind, int ndim, int pos);
    int get_link_ind(int link);
    int get_sc_ind(int sc);
    double get_light_travel_time(double t, int link);
    Vec get_pos(double t, int sc);
};

#endif // __DETECTOR_HPP__