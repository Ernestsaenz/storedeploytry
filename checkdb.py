def get_document_count(self):
  """Get the number of documents in the collection"""
  if self.db:
      collection = self.db.get()
      return len(collection['ids'])
  return 0