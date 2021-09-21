# VAE-GAN
Proyecto de TFG del Grado en Ciencia de Datos. Pre-entrenamiento de GANs con AutoEncoders.

## Organización del GitHub

Para una mejor organización del trabajo, he creado varias carpetas para poder diferenciar las diferentes partes del trabajo.

- **Data:** Es donde subiré los dataset que haya tenido que conseguir de webs no públicas. Por ejemplo, no estará el dataset de CelebA porque cualquiera puede acceder a él y no es necesario.

- **Notebooks:** Es donde estarán todos los notebooks generados a lo largo del trabajo. Ahora mismo cuenta con los siguientes notebooks:

    - **AutoEncoder_densas:** Primera aproximación a los AE a partir de redes densas para comprender su funcionamiento.
    - **AutoEncoder_conv:** Siguiente aproximación con capas convolucionales.
    - **AutoEncoder_conv_colors:** Generalización de la clase para imágenes en color.
    - **AutoEncoderVariacional:** Clase definitiva con el AE variacional que utilizaremos para el pre-entrenamiento de la GAN.
    - **GAN:** Generación de la clase definitva de la GAN con el VAE pre-entrenado.

- **Images:** Imágenes para ejemplificar todo lo que se explique que irán también en el documento final.

- **Scripts:** Estoy usando una herramienta nueva, wandb, para *trackear* experimentos. Para hacerlo más legible, ese seguimiento se hace desde scripts de python después de comprobar en un notebook que la red neuronal funciona correctamente.

- **Papers:** Subiré todos los papers en los que me vaya basando y que posteriormente aparecerán en el apartado *Referencias* del documento final. En la misma carpeta hay otro README donde iré poniendo un pequeño resumen de lo que he entendido y su utilización en el trabajo.

## Planteamiento
La idea que plantea este proyecto es el asegurar y facilitar la convergencia de las redes adversativas evitando problemáticas como las del *modo colapso*. Una GAN tiene un diagrama como el siguiente:

![alt text](/Images/GAN1.PNG "Diagrama GAN")

Entrenar una red adversativa es un reto complicado por las innumerables complicaciones que presenta. Una de ellas es poder establecer un ritmo de aprendizaje parejo entre discriminador y generador. Podría darse el caso, y es bastante común, que el discriminador aprendiera a hacer su trabajo mucho antes de que el genereador estuviera listo para crear imágenes que engañaran al clasificador. Siguiendo esta línea argumental, hay muchas publicaciones e investigaciones que intentan mitigar el problema con soluciones muy variadas. Una de ellas, es el paper en el que nos vamos a centrar para este TFG: La idea de pre-entrenar el generado de la GAN a través de un autoencoder variacional. Un VAE tiene un diagrama como el siguiente:

![alt text](/Images/VAE.png "Diagrama VAE")











