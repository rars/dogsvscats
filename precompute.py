class FeatureProvider(object):
  def __init__(self, data_provider):
    self._data_provider = data_provider

  def get_features(self, model, batches, filename):
    print('Attempting to load data from {0}'.format(filename))
    data = self._data_provider.load_array(filename)
    if data is None:
      print('Data for {0} not available'.format(filename))
      print('Getting data for {0}'.format(filename))
      data = model.predict_generator(batches, batches.nb_sample)
      print('Saving data to {0}'.format(filename))
      data_provider.save_array(filename, data)
      print('Saved data for {0}'.format(filename))
    return data
