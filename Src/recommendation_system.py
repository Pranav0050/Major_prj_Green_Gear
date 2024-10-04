import numpy as np

class SimpleHybridRecommendationSystem:
    def __init__(self, cars, user_ratings):
        self.cars = cars
        self.user_ratings = user_ratings

    def get_recommendations(self, user_id, selected_car_id, top_n=5):
        user_id = int(user_id)
        selected_car_id = int(selected_car_id)
        
        # Collaborative filtering
        user_similarities = np.dot(self.user_ratings, self.user_ratings[user_id]) / (np.linalg.norm(self.user_ratings, axis=1) * np.linalg.norm(self.user_ratings[user_id]))
        
        # Content-based filtering
        selected_car = next(car for car in self.cars if car['id'] == selected_car_id)
        content_scores = [self._calculate_similarity(car, selected_car) for car in self.cars]
        
        # Combine scores
        combined_scores = 0.7 * user_similarities + 0.3 * np.array(content_scores)
        
        # Get top N recommendations
        top_indices = combined_scores.argsort()[-top_n-1:][::-1]
        recommendations = [self.cars[i] for i in top_indices if self.cars[i]['id'] != selected_car_id][:top_n]
        return recommendations

    def _calculate_similarity(self, car1, car2):
        # Simple similarity based on make, model, and fuel type
        similarity = 0
        if car1['make'] == car2['make']:
            similarity += 0.5
        if car1['model'] == car2['model']:
            similarity += 0.3
        if car1['fuelType'] == car2['fuelType']:
            similarity += 0.2
        return similarity

# Example data
cars = [
    {"id": 1, "make": "Tesla", "model": "Model S", "year": 2015, "price": 45000, "mileage": 43000, "image": "https://cdn.usegalileo.ai/sdxl10/c9a629d0-c0d3-4358-8c41-9fb800356eef.png", "fuelType": "Electric", "transmission": "Automatic"},
    {"id": 2, "make": "BMW", "model": "i3", "year": 2017, "price": 20000, "mileage": 38000, "image": "https://cdn.usegalileo.ai/sdxl10/00d173b4-092f-46d1-9f21-b5cc7ca8375c.png", "fuelType": "Electric", "transmission": "Automatic"},
    {"id": 3, "make": "Audi", "model": "A4", "year": 2016, "price": 23000, "mileage": 40000, "image": "https://cdn.usegalileo.ai/sdxl10/304c0799-dfa6-4d06-bdda-1de38ed474ef.png", "fuelType": "Gasoline", "transmission": "Automatic"}
]

user_ratings = np.array([
    [5, 3, 0],
    [4, 0, 5],
    [3, 1, 4],
])