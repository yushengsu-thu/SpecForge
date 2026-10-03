| Algorithm | Configuration | Steady ms/window (range) | Change vs compiled native |
| --- | --- | ---: | ---: |
| dflash | bf16 | 230.62 (230.00–230.69) | +0.00% |
| dflash | cuda | 225.20 (224.59–225.67) | -2.35% |
| dflash | graph-full | 222.66 (222.37–222.83) | -3.45% |
| dflash2 | bf16 | 213.69 (213.68–214.27) | +0.00% |
| dflash2 | cuda | 208.52 (207.87–208.71) | -2.42% |
| dflash2 | graph-full | 212.87 (212.67–213.04) | -0.38% |
| dspark | bf16 | 419.72 (419.17–419.86) | +0.00% |
| dspark | cuda | 413.14 (413.10–414.77) | -1.57% |
| dspark | graph-full | 306.63 (306.06–306.86) | -26.94% |

| Algorithm | Configuration | Allocated GiB | Reserved GiB | Build s | First window s (range) |
| --- | --- | ---: | ---: | ---: | ---: |
| dflash | bf16 | 13.069 | 14.350 | 35.10 | 2.32 (2.28–11.35) |
| dflash | cuda | 6.089 | 14.420 | 35.52 | 2.32 (2.28–2.37) |
| dflash | graph-full | 7.102 | 16.877 | 35.77 | 6.84 (6.74–37.55) |
| dflash2 | bf16 | 16.235 | 17.783 | 36.68 | 2.67 (2.65–2.69) |
| dflash2 | cuda | 6.806 | 17.748 | 36.66 | 2.80 (2.78–2.80) |
| dflash2 | graph-full | 7.996 | 18.457 | 37.18 | 12.77 (12.69–12.83) |
| dspark | bf16 | 19.199 | 20.721 | 36.94 | 2.44 (2.44–2.44) |
| dspark | cuda | 6.673 | 21.096 | 37.04 | 2.63 (2.60–2.71) |
| dspark | graph-full | 7.832 | 24.721 | 40.24 | 36.06 (35.84–44.01) |
