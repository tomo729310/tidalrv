from jax import jit

@jit
def lgcoeffs(wavnm):
    x = (wavnm - 600.)/200.
    fy = lambda x: (((0.06690458*x - 0.15430695)*x + 0.09762098)*x - 0.13098096)*x + 0.54212228
    fu1 = lambda x: (((-0.02030541*x - 0.0565222)*x + 0.16281937)*x - 0.29631237)*x + 0.61250245
    fu2 = lambda x: (((0.02157778*x + 0.07181101)*x - 0.18479906)*x + 0.18314384)*x + 0.13359151
    return fy(x), fu1(x), fu2(x)
