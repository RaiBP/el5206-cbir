# EL5206 CBIR
Content Based Image Retrieval (CBIR) project based on the INRIA Holidays dataset, for the course EL5026 Computational Intelligence and Robotics Laboratory, at the University of Chile.

# How to use
      $ python main.py [-h] [-db] [-di] [-cl] [-cldb] [--distance {euclidean,cosine}] N {hog,resnet}

      Get 10 similar images.

      positional arguments:
        N                     class of image query (integer between 0 and 499)
        {hog,resnet}          feature extractor, either 'hog' or 'resnet'

      optional arguments:
        -h, --help            show this help message and exit
        -db, --generate_database
                              calculate feature vectors and generate new features database
        -di, --download_images
                              forces a re-download of the image dataset
        -cl, --clustering     use k-means clustering
        -cldb, --clustering_new_database
                              use k-means clustering and force creation of new clustering database
        --distance {euclidean,cosine}
                              similarity metric, either 'euclidean' or 'cosine'
