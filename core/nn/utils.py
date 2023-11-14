"""
MicEMouse Utils
=====
This module contains utility functions and classes:
Provides
  1. CacheMixin: A mixin class that provides caching functionality for Python objects.
"""
import os
import pickle
import hashlib
import inspect
import logging

class CacheMixin:
    """
    A mixin class that adds caching functionality. 
    The created object is cached as a pickle file, and the same object is retrieved from cache 
    if created again with the same arguments.
    """

    @classmethod
    def cache_constructor(cls, *args, **kwargs):
        """
        A class method that either creates a new object if it does not exist in the cache, 
        or retrieves it from the cache if it already exists.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            obj: An instance of the class.
        """

        # Get the source code of the class file
        source_code = inspect.getsource(cls)

        str_to_hash =  f'{str(source_code)}_{"_".join(map(str, args))}_{"_".join(map(str, kwargs.values()))}'

        # Compute the MD5 hash of the source code and the argument list
        md5 = hashlib.md5(str_to_hash.encode()).hexdigest()
        logging.debug(f"MD5 hash: {md5}")

        # Create a name for the pickle file using the MD5 hash and the argument list
        file_name = f'.cache/{md5}.pickle'

        os.makedirs('.cache', exist_ok=True)

        # If the pickle file exists, load the object from the pickle file
        if os.path.isfile(file_name):
            logging.info(f"Loading object from cache: {file_name}")
            with open(file_name, 'rb') as file:
                obj = pickle.load(file)
            return obj

        # If the pickle file does not exist, create a new object, pickle it, and save it to a file
        else:
            logging.info(f"Creating new object and saving it to cache: {file_name}")
            obj = cls(*args, **kwargs)
            with open(file_name, 'wb') as file:
                pickle.dump(obj, file)
            return obj

    @staticmethod
    def save_object(obj, file_name):
        """
        A method that pickles and saves an object to a file.

        Args:
            obj: The object to be saved.
            file_name: The name of the file to save the object in.
        """

        with open(file_name, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def load_object(file_name):
        """
        A method that loads a pickled object from a file.

        Args:
            file_name: The name of the file to load the object from.

        Returns:
            The loaded object.
        """

        with open(file_name, 'rb') as file:
            return pickle.load(file)
