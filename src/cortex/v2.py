"""
System Classification: src.cortex.v2
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Theoretical Framework:
    - SubstanceNet theoretical framework (Onasenko, 2025-2026)
    - Visual Cortex V2 Stripes (Hubel & Wiesel, 1968)

Visual Cortex V2 — Contour and Texture Processing (MosaicField18)
===========================================================
Implements V2 stripe architecture based on Hubel & Wiesel's
discovery of three functionally distinct stripe types in V2:

    - Thick stripes:  Motion/change detection (temporal differences)
    - Thin stripes:   Texture/frequency analysis (spectral domain)  
    - Pale stripes:   Form/contour preservation (direct pathway)

This module is CRITICAL for preventing abstract collapse and
consciousness saturation. The three parallel processing pathways
create representations that cannot trivially converge, ensuring
diverse input to the AbstractionLayer and ReflexiveConsciousness.

Mathematical Basis:
    thick(x) = Linear(roll(x, 1) - x)      — temporal gradient
    thin(x)  = Linear(ReLU(FFT(x).real))    — spectral features
    pale(x)  = Linear(x)                    — identity pathway
    V2(x)    = cat([thick, thin, pale])     — multi-stream fusion

Empirical Evidence:
    v3.1.1 WITH MosaicField18:    R = 0.382 (optimal range)
    v4 WITHOUT MosaicField18:     R = 0.999 (saturated)

Key References:
    - Hubel D.H., Wiesel T.N. (1968) J. Physiol. 195:215-243
    - Livingstone M.S., Hubel D.H. (1988) Science 240:740-749

Changelog:
    2026-03-15 v1.0.0 — Ported from v3.1.1 MosaicField18, documented as V2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MosaicField18(nn.Module):
    """
    V2 cortex: three-stripe architecture for contour/texture processing.

    Implements the three functionally distinct stripe types discovered
    by Hubel & Wiesel in visual cortex area V2. Each stripe type
    processes the same input through a different computational pathway,
    creating multi-faceted representations that resist collapse.

    Parameters
    ----------
    in_channels : int
        Input feature dimensionality.
    out_channels : int
        Output feature dimensionality (split ~equally among 3 stripes).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        third = out_channels // 3
        remainder = out_channels - 2 * third

        # Thick stripes: motion/change detection via temporal differences
        self.thick_stripes = nn.Linear(in_channels, third)

        # Thin stripes: texture analysis via frequency domain (FFT)
        self.thin_stripes = nn.Linear(in_channels, third)

        # Pale stripes: form/contour via direct pathway
        self.pale_stripes = nn.Linear(in_channels, remainder)

    def forward(self, x: torch.Tensor, return_streams: bool = False):
        """
        Three-pathway V2 processing.

        Parameters
        ----------
        x : torch.Tensor
            Features [B, seq_len, in_channels].
        return_streams : bool
            If True, return dict of separate streams for V3 phase interference.
            If False, return concatenated tensor (backward compatible).

        Returns
        -------
        torch.Tensor or tuple(torch.Tensor, dict)
            If return_streams=False: V2 features [B, seq_len, out_channels].
            If return_streams=True: (concatenated, {'thick': motion/change,
                                                    'thin': texture/frequency,
                                                    'pale': form/contour}).
        """
        # Thick stripes: detect changes between adjacent positions
        # Biological: motion-sensitive neurons, magnocellular pathway
        thick_out = self.thick_stripes(torch.roll(x, shifts=1, dims=1) - x)

        # Thin stripes: frequency analysis via FFT
        # Biological: color/texture-sensitive neurons, parvocellular pathway
        thin_out = self.thin_stripes(F.relu(torch.fft.fftn(x, dim=1).real))

        # Pale stripes: direct form pathway
        # Biological: form/contour-sensitive neurons, interstripe
        pale_out = self.pale_stripes(x)

        concatenated = torch.cat([thick_out, thin_out, pale_out], dim=2)

        if return_streams:
            streams = {
                'thick': thick_out,   # motion/change detection
                'thin': thin_out,     # texture/frequency analysis
                'pale': pale_out,     # form/contour preservation
            }
            return concatenated, streams
        
        return concatenated
