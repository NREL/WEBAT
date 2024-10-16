import math


class EuclideanDistTracker:
    '''
    This takes the bounding box of the object and save them into one array
    Return all the tracking information needed
    '''
    def __init__(self):
        # Store the center positions of the objects - set with tuple key {(object_type, id) : ((cX, cY), last_frame_num)}
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = {'bat': 1, 'bird': 1, 'insect': 1}         # order: 'bat', 'bird', 'insect'

    def update(self, object_type, cX, cY, frame_num):
        
        # Find out if that object was detected already
        for k, (pt, last_frame_num) in sorted(self.center_points.items(), reverse=True):
            if object_type in k:
                # print(f"[{object_type}] with id {k[1]}")
                dist = math.hypot(cX - pt[0], cY - pt[1])
                if dist < 150 and abs(frame_num-last_frame_num) < 300:      # Acceptable distance and absence time length
                    # Update current centroids with new one
                    existing_id = k[1]
                    self.center_points[(object_type, existing_id)] = ((cX, cY), frame_num)
                    # print("update with existing one..")
                    return existing_id
        
        # New object is detected - we assign the new ID to that object
        new_id = self.id_count[object_type]
        self.center_points[(object_type, new_id)] = ((cX, cY), frame_num)
        self.id_count[object_type] += 1

        # Clean the dictionary by deleting old IDs - maintain with recent 5 IDs for each object type
        if new_id > 5:
            del self.center_points[(object_type, new_id-5)]
        
        # print("The list of centroids:\n", self.center_points)

        return new_id
