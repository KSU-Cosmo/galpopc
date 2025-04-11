#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h> 

// === Gaussian RNG ===
static float randn(unsigned int* seed) {
    float u1 = (float)rand_r(seed) / RAND_MAX;
    float u2 = (float)rand_r(seed) / RAND_MAX;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

// === Central galaxies ===
static PyObject* compute_centrals(PyObject* self, PyObject* args) {
    PyArrayObject *h_mass_log10_arr, *h_z_arr, *h_vel_arr, *h_sigma_arr;
    PyArrayObject *out_z_arr, *out_mask_arr;

    float lnMcut, sigma, alpha_c;
    unsigned int seed;

    if (!PyArg_ParseTuple(args, "O!O!O!O!fffIO!O!",
                          &PyArray_Type, &h_mass_log10_arr,
                          &PyArray_Type, &h_z_arr,
                          &PyArray_Type, &h_vel_arr,
                          &PyArray_Type, &h_sigma_arr,
                          &lnMcut, &sigma, &alpha_c, &seed,
                          &PyArray_Type, &out_z_arr,
                          &PyArray_Type, &out_mask_arr)) {
        return NULL;
    }

    int N = PyArray_SIZE(h_mass_log10_arr);
    float* h_mass_log10 = (float*)PyArray_DATA(h_mass_log10_arr);
    float* h_z = (float*)PyArray_DATA(h_z_arr);
    float* h_vel = (float*)PyArray_DATA(h_vel_arr);
    float* h_sigma = (float*)PyArray_DATA(h_sigma_arr);
    float* out_z = (float*)PyArray_DATA(out_z_arr);
    uint8_t* out_mask = (uint8_t*)PyArray_DATA(out_mask_arr);

    float inv_sqrt2sigma = 1.0f / (sqrtf(2.0f) * sigma);
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float x = (lnMcut - h_mass_log10[i]) * inv_sqrt2sigma;
        float p = 0.5f * erfcf(x);
        unsigned int tid_seed = seed + i + 137 * omp_get_thread_num();
        float r = (float)rand_r(&tid_seed) / RAND_MAX;
        float g = randn(&tid_seed);

        if (r < p) {
            out_mask[i] = 1;
            float z = h_z[i];
            if (rsd) {
                z += h_vel[i] + alpha_c * h_sigma[i] * g;
                if (z > Lmax) z -= Lbox;
                if (z < Lmin) z += Lbox;
            }
            out_z[i] = z;
        } else {
            out_mask[i] = 0;
            out_z[i] = h_z[i];
        }
        
    }

    Py_RETURN_NONE;
}

static PyObject* compute_satellites(PyObject* self, PyObject* args) {
    PyArrayObject *s_mass_log10_arr, *s_mass_arr, *s_npart_arr;
    PyArrayObject *s_z_arr, *s_vel_arr, *s_host_vel_arr;
    PyArrayObject *out_z_arr, *out_mask_arr;

    float lnMcut, sigma, M1, kappa, alpha, alpha_s, Lmin, Lmax;
    int rsd;
    unsigned int seed;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!ffffffffIIO!O!",
                          &PyArray_Type, &s_mass_log10_arr,
                          &PyArray_Type, &s_mass_arr,
                          &PyArray_Type, &s_npart_arr,
                          &PyArray_Type, &s_z_arr,
                          &PyArray_Type, &s_vel_arr,
                          &PyArray_Type, &s_host_vel_arr,
                          &lnMcut, &sigma, &M1, &kappa, &alpha, &alpha_s,
                          &Lmin, &Lmax, &rsd, &seed,
                          &PyArray_Type, &out_z_arr,
                          &PyArray_Type, &out_mask_arr)) {
        return NULL;
    }

    int N = PyArray_SIZE(s_mass_log10_arr);
    float* s_mass_log10 = (float*)PyArray_DATA(s_mass_log10_arr);
    float* s_mass = (float*)PyArray_DATA(s_mass_arr);
    float* s_npart = (float*)PyArray_DATA(s_npart_arr);
    float* s_z = (float*)PyArray_DATA(s_z_arr);
    float* s_vel = (float*)PyArray_DATA(s_vel_arr);
    float* s_host_vel = (float*)PyArray_DATA(s_host_vel_arr);
    float* out_z = (float*)PyArray_DATA(out_z_arr);
    uint8_t* out_mask = (uint8_t*)PyArray_DATA(out_mask_arr);

    float inv_sqrt2sigma = 1.0f / (sqrtf(2.0f) * sigma);
    float Lbox = Lmax - Lmin;
    float Mcut = powf(10.0f, lnMcut);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        unsigned int tid_seed = seed + i + 137 * omp_get_thread_num();  // vary seed across threads

        float x = (lnMcut - s_mass_log10[i]) * inv_sqrt2sigma;
        float ncen = 0.5f * erfcf(x);

        float m_eff = s_mass[i] - kappa * Mcut;
        if (m_eff <= 0.0f) {
            out_mask[i] = 0;
            out_z[i] = s_z[i];
            continue;
        }

        float mrat = m_eff / M1;
        float nsat = expf(alpha * logf(mrat)) * ncen;
        float psat = nsat / s_npart[i];
        if (psat > 1.0f) psat = 1.0f;
        if (psat < 0.0f) psat = 0.0f;

        float r = (float)rand_r(&tid_seed) / RAND_MAX;

        if (r < psat) {
            out_mask[i] = 1;
            float z = s_z[i];
            if (rsd) {
                z += s_host_vel[i] + alpha_s * (s_vel[i] - s_host_vel[i]);
                if (z > Lmax) z -= Lbox;
                if (z < Lmin) z += Lbox;
            }
            out_z[i] = z;
        } else {
            out_mask[i] = 0;
            out_z[i] = s_z[i];
        }
    }

    Py_RETURN_NONE;
}

// === Python module setup ===
static PyMethodDef methods[] = {
    {"compute_centrals", compute_centrals, METH_VARARGS, "Compute central galaxy mask and RSD"},
    {"compute_satellites", compute_satellites, METH_VARARGS, "Compute satellite mask and RSD"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "galcore",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_galcore(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
