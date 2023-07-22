# aai-selfdriving-cars

## Information about Data Sets
Here are some use full Information about creating own datasets.
We can use these information to create a data set for our own purpose.
The structure of the dataset, described here, amies a uncompilcated way to train YOLOv8 models.

### Folder structure

./"Name" dataset
    images/
        test
        train
        valid

### Data Tips and Tricks
* 20% valid - 80% train data or 30% - 70% 
* ensure a good diversity and variance (no sequences a video clip as frames, because there are heavily correlated)
* to care of luminosity and different angles much more data have to be used
* no overlapping between training and validation data
* amount of data for different classes should be balanced (200 car pictures, 200 truck pictures)
