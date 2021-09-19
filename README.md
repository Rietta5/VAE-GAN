# VAE-GAN
Proyecto de TFG del Grado en Ciencia de Datos. Pre-entrenamiento de GANs con AutoEncoders.

## Organización del GitHub

Para una mejor organización del trabajo, he creado varias carpetas para poder diferenciar las diferentes partes del trabajo.

- **Data:** Es donde subiré los dataset que haya tenido que conseguir de webs no públicas. Por ejemplo, no estará el dataset de CelebA porque cualquiera puede accder a él y no es necesario.

- **Notebooks:** Es donde estarán todos los notebooks generados a lo largo del trabajo. Ahora mismo cuenta con los siguientes notebooks:

    - **AutoEncoder_densas:** Primera aproximación a los AE a partir de redes densas para comprender su funcionamiento.
    - **AutoEncoder_conv:** Siguiente aproximación con capas convolucionales.
    - **AutoEncoder_conv_colors:** Generalización de la clase para imágenes en color.
    - **AutoEncoder_var:** Clase definitiva con el AE variacional que utilizaremos para el pre-entrenamiento de la GAN.

- **Papers:** Subiré todos los papers en los que me vaya basando y que posteriormente aparecerán en el apartado *Referencias* del documento final. En la misma carpeta hay otro README donde iré poniendo un pequeño resumen de lo que he entendido y su utilización en el trabajo.

## Planteamiento

El trabajo consiste en 