# TRIPS and the International Geomagnetic Reference Field (IGRF)


TRIPS comes bundled with a coefficient file for IGRF 13. The coefficient file path is specified to TRIPS in plaintext within another file at `trips_lstar/data/specify_IGRFfile.json`. To load a different coefficient file, the user should
- find where TRIPS is installed, from a Python interpreter run: `import trips_lstar;print(trips_lstar.__file__)`
- place the new `.shc` file in the `.../trips_lstar/data` directory
- modify `specify_IGRFfile.json` with the new file's name
- re-import trips_lstar


A selection of `.shc` files are available from: [https://github.com/ESA-VirES/MagneticModel/tree/staging/eoxmagmod/eoxmagmod/data](https://github.com/ESA-VirES/MagneticModel/tree/staging/eoxmagmod/eoxmagmod/data). **Note: TRIPS has only been tested using the bundled `IGRF13.shc`**

The bundled IGRF coefficients file was obtained from the Pure Python IGRF repository on Github, available at: [https://github.com/IAGA-VMOD/ppigrf](https://github.com/IAGA-VMOD/ppigrf).