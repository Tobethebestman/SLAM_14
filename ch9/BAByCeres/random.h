#ifndef RAND_H
#define RAND_H

#include <math.h>
#include <stdlib.h>

// rand in [0, 1]
inline double randDouble()
{
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

// rand ~N(0, 1)
inline double randNormal()
{
    double x1, x2, w;
    do{
        x1 = 2.0 * randDouble() - 1.0;
        x2 = 2.0 * randDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 || w == 0.0 );

    w = sqrt((-2.0 * log(w)) / w);
    return x1 * w;
}

#endif