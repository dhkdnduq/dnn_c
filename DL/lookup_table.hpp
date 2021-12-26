#include <list>
namespace messaging {
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class lookup_table {
 private:
  class bucket_type {
   private:
    typedef std::pair<Key, Value> bucket_value;
    typedef std::list<bucket_value> bucket_data;
    typedef typename bucket_data::iterator bucket_iterator;

    bucket_data data;
    mutable std::mutex mutex;
    bucket_iterator find_entry_for(Key const& key) {
      return std::find_if(
          data.begin(), data.end(),
          [&](bucket_value const& item) { return item.first == key; });
    }

   public:
    Value value_for(Key const& key, Value const& default_value) {
      std::unique_lock<std::mutex> lock(mutex);
      bucket_iterator const found_entry = find_entry_for(key);
      return (found_entry == data.end()) ? default_value : found_entry->second;
    }

    bool find_mapping(Key const& key) {
      std::unique_lock<std::mutex> lock(mutex);
      bucket_iterator const found_entry = find_entry_for(key);
      return (found_entry == data.end()) ? false : true;
    }
    void add_or_update_mapping(Key const& key, Value const& value) {
      std::unique_lock<std::mutex> lock(mutex);
      bucket_iterator const found_entry = find_entry_for(key);
      if (found_entry == data.end()) {
        data.push_back(bucket_value(key, value));
      } else {
        found_entry->second = value;
      }
    }

    void remove_mapping(Key const& key) {
      std::unique_lock<std::mutex> lock(mutex);
      bucket_iterator const found_entry = find_entry_for(key);
      if (found_entry != data.end()) {
        data.erase(found_entry);
      }
    }
    bucket_data& get_bucket_list() { return data; }
  };
  std::vector<std::unique_ptr<bucket_type>> buckets;
  Hash hasher;

  bucket_type& get_bucket(Key const& key) const {
    std::size_t const bucket_index = hasher(key) % buckets.size();
    return *buckets[bucket_index];
  }

 public:
  typedef Key key_type;
  typedef Value mapped_type;
  typedef Hash hash_type;

  lookup_table(unsigned num_buckets = 4, Hash const& hasher_ = Hash())
      : buckets(num_buckets), hasher(hasher_) {
    for (unsigned i = 0; i < num_buckets; ++i) {
      buckets[i].reset(new bucket_type());
    }
  }

  Value value_for(Key const& key, Value const& default_value = Value()) const {
    return get_bucket(key).value_for(key, default_value);
  }

  void add_or_update_mapping(Key const& key, Value const& value) {
    get_bucket(key).add_or_update_mapping(key, value);
  }

  bool find_mapping(Key const& key) {
    return get_bucket(key).find_mapping(key);
  }
  void remove_mapping(Key const& key) { get_bucket(key).remove_mapping(key); }

  std::vector<std::unique_ptr<bucket_type>>& get_bucket() { return buckets; }
};
}  // namespace messaging