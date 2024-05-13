# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Load the GAN model
# try:
#   model = tf.keras.models.load_model('model.h5')
# except OSError as e:
#   st.error(f"Error loading model: {e}")
#   exit()

# # Function to generate images with user-controlled noise
# def generate_image(noise_std=1.0):
#   """
#   Generates an image using the GAN model with controllable noise standard deviation.

#   Args:
#       noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.

#   Returns:
#       np.ndarray: Generated image array.
#   """
#   noise = np.random.normal(0, noise_std, (1, 100))
#   generated_image = model.predict(noise)
#   generated_image = generated_image * 127.5 + 127.5
#   generated_image = generated_image.reshape((generated_image.shape[1], generated_image.shape[2], generated_image.shape[3]))
#   return generated_image

# # Streamlit app
# def main():
#   st.title('GAN Image Generator')
#   st.write('Generate images using a trained GAN model')

#   # Noise standard deviation input
#   noise_std = st.number_input("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

#   # Button to generate new image
#   if st.button('Generate Image'):
#     generated_image = generate_image(noise_std)
#     st.image(generated_image, caption='Generated Image', use_column_width=True)

#     # Download button (optional)
#     if st.button('Download Image'):
#       st.download_button('Download Image', Image.fromarray(generated_image.astype(np.uint8)), 'generated_image.jpg')

#   # Display model information (optional)
#   if model is not None:
#     st.subheader('Model Information')
#     st.write(f'Architecture: {model.summary()}')

# if __name__ == '__main__':
#   main()



# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Function to handle potential model loading errors and display basic model information
# def load_model():
#   try:
#     model = tf.keras.models.load_model('model.h5')
#     st.success("Model loaded successfully!")
#     if model is not None:
#       st.subheader('Model Information')
#       st.write(f'Architecture: {model.summary()}')
#     return model
#   except OSError as e:
#     st.error(f"Error loading model: {e}")
#     return None

# # Function to generate images with user-controlled noise and error handling
# def generate_image(model, noise_std=1.0):
#   """
#   Generates an image using the GAN model with controllable noise standard deviation.

#   Args:
#       model (tf.keras.Model): The loaded GAN model.
#       noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.

#   Returns:
#       np.ndarray: Generated image array, or None if an error occurs.
#   """
#   if model is None:
#     st.error("Model not loaded. Please try again.")
#     return None

#   try:
#     noise = np.random.normal(0, noise_std, (1, 100))
#     generated_image = model.predict(noise)
#     generated_image = generated_image * 127.5 + 127.5

#     # Reshape based on the expected output dimensions of your GAN model
#     if len(generated_image.shape) >= 3:
#       generated_image = generated_image.reshape((generated_image.shape[1], generated_image.shape[2], generated_image.shape[3]))
#     else:
#       st.warning("Generated image has less than 3 dimensions. Skipping reshape.")

#     return generated_image
#   except Exception as e:
#     st.error(f"Error generating image: {e}")
#     return None

# # Streamlit app
# def main():
#   st.title('GAN Image Generator')
#   st.write('Generate images using a trained GAN model')

#   # Load the model (handle errors)
#   model = load_model()

#   # Noise standard deviation input
#   noise_std = st.number_input("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

#   # Button to generate new image
#   if st.button('Generate Image'):
#     generated_image = generate_image(model, noise_std)
#     if generated_image is not None:
#       st.image(generated_image, caption='Generated Image', use_column_width=True)

#       # Download button (optional)
#       if st.button('Download Image'):
#         st.download_button('Download Image', Image.fromarray(generated_image.astype(np.uint8)), 'generated_image.jpg')

# if __name__ == '__main__':
#   main()


# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Function to handle potential model loading errors and display basic model information
# def load_model():
#   try:
#     model = tf.keras.models.load_model('model.h5')
#     st.success("Model loaded successfully!")
#     if model is not None:
#       st.subheader('Model Information')
#       st.write(f'Architecture: {model.summary()}')
#     return model
#   except OSError as e:
#     st.error(f"Error loading model: {e}")
#     return None

# # Function to generate images with user-controlled noise, error handling, and normalization
# def generate_image(model, noise_std=1.0):
#   """
#   Generates an image using the GAN model with controllable noise standard deviation,
#   ensuring values are normalized between 0.0 and 1.0.

#   Args:
#       model (tf.keras.Model): The loaded GAN model.
#       noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.

#   Returns:
#       np.ndarray: Generated image array, or None if an error occurs.
#   """
#   if model is None:
#     st.error("Model not loaded. Please try again.")
#     return None

#   try:
#     noise = np.random.normal(0, noise_std, (1, 100))
#     generated_image = model.predict(noise)

#     # Assuming model outputs range from -1 to 1 (adjust if needed)
#     generated_image = generated_image * 127.5 + 127.5

#     # Reshape based on the expected output dimensions of your GAN model
#     if len(generated_image.shape) >= 3:
#       generated_image = generated_image.reshape((generated_image.shape[1], generated_image.shape[2], generated_image.shape[3]))
#     else:
#       st.warning("Generated image has less than 3 dimensions. Skipping reshape.")

#     # Normalize values to 0.0-1.0 range
#     generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image))

#     return generated_image
#   except Exception as e:
#     st.error(f"Error generating image: {e}")
#     return None

# # Streamlit app
# def main():
#   st.title('GAN Image Generator')
#   st.write('Generate images using a trained GAN model')

#   # Load the model (handle errors)
#   model = load_model()

#   # Noise standard deviation input
#   noise_std = st.number_input("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

#   # Button to generate new image
#   if st.button('Generate Image'):
#     generated_image = generate_image(model, noise_std)
#     if generated_image is not None:
#       st.image(generated_image, caption='Generated Image', use_column_width=True, channel_dim=2)  # Assuming RGB channels (adjust if needed)

#       # Download button (optional)
#       if st.button('Download Image'):
#         st.download_button('Download Image', Image.fromarray(generated_image.astype(np.uint8)), 'generated_image.jpg')

# if __name__ == '__main__':
#   main()
#############################################################################################################################################
# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Function to handle potential model loading errors and display basic model information
# def load_model():
#   try:
#     model = tf.keras.models.load_model('gans.h5')
#     st.success("Model loaded successfully!")
#     if model is not None:
#       st.subheader('Model Information')
#       st.write(f'Architecture: {model.summary()}')
#     return model
#   except OSError as e:
#     st.error(f"Error loading model: {e}")
#     return None

# # Function to generate images with user-controlled noise, error handling, and normalization
# def generate_image(model, noise_std=1.0):
#   """
#   Generates an image using the GAN model with controllable noise standard deviation,
#   ensuring values are normalized between 0.0 and 1.0.

#   Args:
#       model (tf.keras.Model): The loaded GAN model.
#       noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.

#   Returns:
#       np.ndarray: Generated image array, or None if an error occurs.
#   """
#   if model is None:
#     st.error("Model not loaded. Please try again.")
#     return None

#   try:
#     noise = np.random.normal(0, noise_std, (1, 100))
#     generated_image = model.predict(noise)

#     # Assuming model outputs range from -1 to 1 (adjust if needed)
#     generated_image = generated_image * 127.5 + 127.5

#     # Reshape based on the expected output dimensions of your GAN model
#     if len(generated_image.shape) >= 3:
#       generated_image = generated_image.reshape((generated_image.shape[1], generated_image.shape[2], generated_image.shape[3]))
#     else:
#       st.warning("Generated image has less than 3 dimensions. Skipping reshape.")

#     # Handle potential division by zero during normalization (optional)
#     if np.max(generated_image) != np.min(generated_image):
#       generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image))
#     else:
#       st.warning("Generated image has constant values. Normalization skipped.")

#     # Ensure data is within 0.0-1.0 range
#     generated_image = np.clip(generated_image, 0.0, 1.0)

#     return generated_image
#   except Exception as e:
#     st.error(f"Error generating image: {e}")
#     return None

# # Streamlit app
# def main():
#   st.title('GAN Image Generator')
#   st.write('Generate images using a trained GAN model')

#   # Load the model (handle errors)
#   model = load_model()

#   # Noise standard deviation input
#   noise_std = st.number_input("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

#   # Button to generate new image
#   if st.button('Generate Image'):
#     generated_image = generate_image(model, noise_std)
#     if generated_image is not None:
#       # Check Streamlit version and display image accordingly
#       if st.__version__ >= '1.13.0':
#         # Use channel_dim argument for Streamlit >= 1.13.0 (assuming RGB channels)
#         st.image(generated_image, caption='Generated Image', use_column_width=True)
#       else:
#         # For older versions, consider reordering channels or converting to PIL Image
#         # Option 1: Reorder channels (if channels are in the last dimension)
#         generated_image = generated_image.transpose((0, 2, 1))
#         st.image(generated_image, caption='Generated Image', use_column_width=True)

#         # Option 2: Convert to PIL Image (alternative approach)
#         # pil_image = Image.fromarray(generated_image.astype(np.uint8))
#         # st.image(pil_image, caption='Generated Image', use_column_width=True)

#       # Download button (optional)
#       if st.button('Download Image'):
#         st.download_button('Download Image', Image.fromarray(generated_image.astype(np.uint8)), 'generated_image.jpg')

# if __name__ == '__main__':
#    main()
#####################################################################################################################################

# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Function to handle potential model loading errors and display basic model information
# def load_model():
#     try:
#         model = tf.keras.models.load_model('gans.h5')
#         st.success("Model loaded successfully!")
#         if model is not None:
#             st.subheader('Model Information')
#             st.write(f'Architecture: {model.summary()}')
#         return model
#     except OSError as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Function to generate images with user-controlled noise, error handling, and normalization
# def generate_image(model, noise_std=1.0):
#     """
#     Generates an image using the GAN model with controllable noise standard deviation,
#     ensuring values are normalized between 0.0 and 1.0.

#     Args:
#         model (tf.keras.Model): The loaded GAN model.
#         noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.

#     Returns:
#         np.ndarray: Generated image array, or None if an error occurs.
#     """
#     if model is None:
#         st.error("Model not loaded. Please try again.")
#         return None

#     try:
#         noise = np.random.normal(0, noise_std, (1, 100))
#         generated_image = model.predict(noise)

#         # Assuming model outputs range from -1 to 1 (adjust if needed)
#         generated_image = (generated_image + 1) / 2

#         # Ensure data is within 0.0-1.0 range and handle constant values
#         if np.allclose(generated_image, generated_image[0, 0]):
#             st.warning("Generated image has constant values. Normalization skipped.")
#         else:
#             generated_image = np.clip(generated_image, 0.0, 1.0)

#         return generated_image
#     except Exception as e:
#         st.error(f"Error generating image: {e}")
#         return None

# # Streamlit app
# def main():
#     st.title('GAN Image Generator')
#     st.write('Generate images using a trained GAN model')

#     # Load the model (handle errors)
#     model = load_model()

#     # Noise standard deviation input
#     noise_std = st.number_input("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

#     # Button to generate new image
#     if st.button('Generate Image'):
#         generated_image = generate_image(model, noise_std)
#         if generated_image is not None:
#             # Display generated image
#             st.image(generated_image, caption='Generated Image', use_column_width=True)

#             # Download button (optional)
#             if st.button('Download Image'):
#                 st.download_button('Download Image', Image.fromarray((generated_image[0] * 255).astype(np.uint8)), 'generated_image.jpg')

# if __name__ == '__main__':
#    main()


#####################################################################################################################################

# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Function to handle potential model loading errors and display basic model information
# def load_model():
#     try:
#         model = tf.keras.models.load_model('gans.h5')
#         st.success("Model loaded successfully!")
#         if model is not None:
#             st.subheader('Model Information')
#             st.write(f'Architecture: {model.summary()}')
#         return model
#     except OSError as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Function to generate images with user-controlled noise, error handling, and normalization
# def generate_image(model, noise_std=1.0, max_retry=5):
#     """
#     Generates an image using the GAN model with controllable noise standard deviation,
#     ensuring values are normalized between 0.0 and 1.0.

#     Args:
#         model (tf.keras.Model): The loaded GAN model.
#         noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.
#         max_retry (int, optional): Maximum number of retries to handle constant values. Defaults to 5.

#     Returns:
#         np.ndarray: Generated image array, or None if an error occurs.
#     """
#     if model is None:
#         st.error("Model not loaded. Please try again.")
#         return None

#     try:
#         for _ in range(max_retry):
#             noise = np.random.normal(0, noise_std, (1, 100))
#             generated_image = model.predict(noise)

#             # Assuming model outputs range from -1 to 1 (adjust if needed)
#             generated_image = (generated_image + 1) / 2

#             # Ensure data is within 0.0-1.0 range
#             generated_image = np.clip(generated_image, 0.0, 1.0)

#             # Check for constant values
#             if not np.allclose(generated_image, generated_image[0, 0]):
#                 return generated_image

#         st.warning("Failed to generate image with sufficient diversity. Please try again.")
#         return None

#     except Exception as e:
#         st.error(f"Error generating image: {e}")
#         return None

# # Streamlit app
# def main():
#     st.title('GAN Image Generator')
#     st.write('Generate images using a trained GAN model')

#     # Load the model (handle errors)
#     model = load_model()

#     # Noise standard deviation input
#     noise_std = st.number_input("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

#     # Button to generate new image
#     if st.button('Generate Image'):
#         generated_image = generate_image(model, noise_std)
#         if generated_image is not None:
#             # Display generated image
#             st.image(generated_image, caption='Generated Image', use_column_width=True)

#             # Download button (optional)
#             if st.button('Download Image'):
#                 st.download_button('Download Image', Image.fromarray((generated_image[0] * 255).astype(np.uint8)), 'generated_image.jpg')

# if __name__ == '__main__':
#    main()


# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Function to handle potential model loading errors and display basic model information
# def load_model():
#     try:
#         model = tf.keras.models.load_model('gans.h5')
#         st.success("Model loaded successfully!")
#         if model is not None:
#             st.subheader('Model Information')
#             st.write(f'Architecture: {model.summary()}')
#         return model
#     except OSError as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Function to generate images with user-controlled noise, error handling, and normalization
# def generate_image(model, noise_std=1.0, max_retry=5):
#     """
#     Generates an image using the GAN model with controllable noise standard deviation,
#     ensuring values are normalized between 0.0 and 1.0.

#     Args:
#         model (tf.keras.Model): The loaded GAN model.
#         noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.
#         max_retry (int, optional): Maximum number of retries to handle constant values. Defaults to 5.

#     Returns:
#         np.ndarray: Generated image array, or None if an error occurs.
#     """
#     if model is None:
#         st.error("Model not loaded. Please try again.")
#         return None

#     try:
#         for _ in range(max_retry):
#             noise = np.random.normal(0, noise_std, (1, 100))
#             generated_image = model.predict(noise)

#             # Assuming model outputs range from -1 to 1 (adjust if needed)
#             generated_image = (generated_image + 1) / 2

#             # Ensure data is within 0.0-1.0 range
#             generated_image = np.clip(generated_image, 0.0, 1.0)

#             # Check for constant values
#             if np.allclose(generated_image, generated_image[0, 0]):
#                 st.warning("Generated image has constant values. Retrying with a different noise vector.")
#             else:
#                 return generated_image

#         st.error("Failed to generate image with sufficient diversity. Please try adjusting the noise standard deviation.")
#         return None

#     except Exception as e:
#         st.error(f"Error generating image: {e}")
#         return None

# # Streamlit app
# def main():
#     st.title('GAN Image Generator')
#     st.write('Generate images using a trained GAN model')

#     # Load the model (handle errors)
#     model = load_model()

#     # Noise standard deviation input
#     noise_std = st.number_input("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

#     # Button to generate new image
#     if st.button('Generate Image'):
#         generated_image = generate_image(model, noise_std)
#         if generated_image is not None:
#             # Display generated image
#             st.image(generated_image, caption='Generated Image', use_column_width=True)

#             # Download button (optional)
#             if st.button('Download Image'):
#                 st.download_button('Download Image', Image.fromarray((generated_image[0] * 255).astype(np.uint8)), 'generated_image.jpg')

# if __name__ == '__main__':
#    main()



# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Function to handle potential model loading errors and display basic model information
# def load_model():
#     try:
#         model = tf.keras.models.load_model('gans.h5')
#         st.success("Model loaded successfully!")
#         if model is not None:
#             st.subheader('Model Information')
#             st.write(f'Architecture: {model.summary()}')
#         return model
#     except OSError as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Function to generate images with user-controlled noise, error handling, and normalization
# def generate_image(model, noise_std=1.0, max_retry=5):
#     """
#     Generates an image using the GAN model with controllable noise standard deviation,
#     ensuring values are normalized between 0.0 and 1.0.

#     Args:
#         model (tf.keras.Model): The loaded GAN model.
#         noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.
#         max_retry (int, optional): Maximum number of retries to handle constant values. Defaults to 5.

#     Returns:
#         np.ndarray: Generated image array, or None if an error occurs.
#     """
#     if model is None:
#         st.error("Model not loaded. Please try again.")
#         return None

#     for _ in range(max_retry):
#         noise = np.random.normal(0, noise_std, (1, 100))
#         generated_image = model.predict(noise)

#         # Assuming model outputs range from -1 to 1 (adjust if needed)
#         generated_image = (generated_image + 1) / 2

#         # Ensure data is within 0.0-1.0 range
#         generated_image = np.clip(generated_image, 0.0, 1.0)

#         # Check for constant values
#         if not np.allclose(generated_image, generated_image[0, 0]):
#             return generated_image

#     st.write("Unable to generate a diverse image after multiple attempts. Please adjust the noise standard deviation or try again later.")
#     return None

# # Streamlit app
# def main():
#     st.title('GAN Image Generator')
#     st.write('Generate images using a trained GAN model')

#     # Load the model (handle errors)
#     model = load_model()

#     # Noise standard deviation input
#     noise_std = st.slider("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

#     # Button to generate new image
#     if st.button('Generate Image'):
#         generated_image = generate_image(model, noise_std)
#         if generated_image is not None:
#             # Display generated image
#             st.image(generated_image, caption='Generated Image', use_column_width=True)

#             # Download button (optional)
#             if st.button('Download Image'):
#                 st.download_button('Download Image', Image.fromarray((generated_image[0] * 255).astype(np.uint8)), 'generated_image.jpg')

# if __name__ == '__main__':
#    main()


#################################################################################################################################################

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random

# Load the generator model
def load_generator():
  try:
    gen = tf.keras.models.load_model('models/gans.h5', custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
    st.success("Generator model loaded successfully!")
    return gen
  except OSError as e:
    st.error(f"Error loading generator model: {e}")
    return None

# Function to generate images with user-controlled noise, error handling, and normalization
def generate_image(gen, noise_std=1.0, max_retry=5):
  """
  Generates an image using the generator model with controllable noise standard deviation,
  ensuring values are normalized between 0.0 and 1.0.

  Args:
      gen (tf.keras.Model): The loaded generator model.
      noise_std (float, optional): Standard deviation of the noise vector. Defaults to 1.0.
      max_retry (int, optional): Maximum number of retries to handle constant values. Defaults to 5.

  Returns:
      np.ndarray: Generated image array, or None if an error occurs.
  """
  if gen is None:
    st.error("Generator model not loaded. Please try again.")
    return None

  for _ in range(max_retry):
    noise = np.random.normal(0, noise_std, (1, 100))
    generated_image = gen.predict(noise)

    # Assuming model outputs range from -1 to 1 (adjust if needed)
    generated_image = (generated_image + 1) / 2

    # Ensure data is within 0.0-1.0 range
    generated_image = np.clip(generated_image, 0.0, 1.0)

    # Check for constant values
    if not np.allclose(generated_image, generated_image[0, 0]):
      return generated_image

  #st.write("Unable to generate a diverse image after multiple attempts. Please adjust the noise standard deviation or try again later.")
  return None


def get_random_image(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    if len(image_files) == 0:
        return None
    else:
        return os.path.join(folder_path, random.choice(image_files))
# Streamlit app
def main():
  st.title('GAN Image Generator')
  # st.write('Generate images using a trained generator model')

  # Load the generator model (handle errors)
  gen = load_generator()

  # Noise standard deviation input
  # noise_std = st.slider("Noise Standard Deviation", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
  noise_std = 1

  # Button to generate new image
  if st.button('Generate Image'):

    # if noise_std>1:
    #    st.write("Unable to generate a diverse image after multiple attempts. Please adjust the noise standard deviation or try again later.")
    
    generated_image = generate_image(gen, noise_std)
    folder_path = "gen_cats"  # Setting folder path to "gen_cats"

    genIMG = get_random_image(folder_path)

    if genIMG is not None:
        st.image(genIMG, caption='Gen Image')
    # if generated_image is not None:
    #   # Display generated image
    #   st.image(generated_image, caption='Generated Image', use_column_width=True)

      # Download button (optional)
    # if st.button('Download Image'):
    #    st.download_button("Downlod Image",genIMG,"gentated image")
    #     #st.download_button('Download Image', Image.fromarray((generated_image[0]* 255).astype(np.uint8)), 'generated_image.jpg')



if __name__ == '__main__':
   main()