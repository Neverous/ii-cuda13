std::string kernelSource = R"(

struct Complex
{
    float x,
         y;
};

inline
struct Complex Complex_(float x, float y)
{
    struct Complex ret;
    ret.x = x;
    ret.y = y;
    return ret;
}

inline
struct Complex complex_sub(struct Complex self, struct Complex sub)
{
    self.x -= sub.x;
    self.y-= sub.y;
    return self;
}

inline
struct Complex complex_add(struct Complex self, struct Complex add)
{
    self.x += add.x;
    self.y += add.y;
    return self;
}

inline
struct Complex complex_mul1(struct Complex self, struct Complex mul)
{
    struct Complex ret;
    ret.x = self.x * mul.x - self.y * mul.y;
    ret.y = self.y * mul.x + self.x * mul.y;
    return ret;
}

inline
struct Complex complex_mul2(struct Complex self, float mul)
{
    self.x *= mul;
    self.y *= mul;
    return self;
}

inline
struct Complex complex_div(struct Complex self, float div)
{
    self.x /= div;
    self.y /= div;
    return self;
}

inline
struct Complex complex_pow(struct Complex self, int _pow)
{
    struct Complex ret = self;
    while(-- _pow > 0)
        ret = complex_mul1(ret, self);

    return ret;
}

inline
struct Complex complex_inv(struct Complex self)
{
    struct Complex ret = complex_div(self, self.x * self.x + self.y * self.y);
    ret.y *= -1;
    return ret;
}

inline
float complex_norm(struct Complex self)
{
    return sqrt(self.x * self.x + self.y * self.y);
}

#define MAX_STEPS 2048

inline
struct Complex newton(struct Complex start)
{
    struct Complex prev = Complex_(0., 0.);
    for(int s = 0; complex_norm(complex_sub(prev, start)) > __FLT_EPSILON__ && s < MAX_STEPS; ++ s)
    {
        struct Complex step = complex_div(complex_add(complex_mul2(start, 3), complex_pow(complex_inv(start), 3)), 4);
        prev = start;
        start = step;
    }

    return start;
}

// OpenCL kernel
kernel void fraktal_opencl(global char *image, int width, int height, float s)
{
    float hstep = 2. * s / height,
          wstep = 2. * s / width;

    int h = get_group_id(0),
        w = get_local_id(0) * width / get_local_size(0),
        end = (get_local_id(0) + 1) * width / get_local_size(0);

    while(w < end)
    {
        float hs = -s + hstep * h,
              ws = -s + wstep * w;

        struct Complex res = newton(Complex_(ws, hs));
        if(complex_norm(complex_sub(res, Complex_(0., 1.))) < __FLT_EPSILON__)
            image[h * width + w] = 2;

        else if(complex_norm(complex_sub(res, Complex_(1., 0.))) < __FLT_EPSILON__)
            image[h * width + w] = 3;

        else if(complex_norm(complex_sub(res, Complex_(-1., 0.))) < __FLT_EPSILON__)
            image[h * width + w] = 4;

        else if(complex_norm(complex_sub(res, Complex_(0., -1.))) < __FLT_EPSILON__)
            image[h * width + w] = 5;

        else
            image[h * width + w] = 1;

        ++ w;
    }
}
)";
