from collections import defaultdict
import numpy as np
import Levenshtein

class BoundingBoxOCR:
    def get_word_center_y(self, bbox):
        """Get the average Y center of a bounding box."""
        return sum([point['y'] for point in bbox]) / len(bbox)

    def merge_bounding_boxes(self, bboxes):
        """Create a bounding box that encloses all given bounding boxes."""
        all_x = [point['x'] for bbox in bboxes for point in bbox]
        all_y = [point['y'] for bbox in bboxes for point in bbox]
        return [
            {"x": min(all_x), "y": min(all_y)},
            {"x": max(all_x), "y": min(all_y)},
            {"x": max(all_x), "y": max(all_y)},
            {"x": min(all_x), "y": max(all_y)}
        ]
        
    def extract_words(self, read_results):
        words = []
        for page_data in read_results:
            page_number = page_data['page']
            for line in page_data['lines']:
                for word in line['words']:
                    bbox = word['boundingPolygon']
                    bounding_polygon = [{"x": bbox[i], "y": bbox[i+1]} for i in range(0, len(bbox), 2)]
                    
                    words.append({
                        "text": word['text'],
                        "boundingPolygon": word['boundingPolygon'],
                        "center_y": self.get_word_center_y(word['boundingPolygon']),
                        "page": page_number
                    })
        return words
    
    def group_words(self, words, y_threshold=10):
        words = sorted(words, key=lambda w: (w['page'], w['center_y']))
        clusters_by_page = defaultdict(list)

        for word in words:
            page = word['page']
            assigned = False
            for cluster in clusters_by_page[page]:
                if abs(word['center_y'] - cluster['avg_y']) <= y_threshold:
                    cluster['words'].append(word)
                    cluster['avg_y'] = np.mean([w['center_y'] for w in cluster['words']])
                    assigned = True
                    break
            if not assigned:
                clusters_by_page[page].append({
                    'words': [word],
                    'avg_y': word['center_y']
                })

        candidates = []
        for page, clusters in clusters_by_page.items():
            for cluster in clusters:
                sorted_words = sorted(cluster['words'], key=lambda w: w['boundingPolygon'][0]['x'])
                candidates.append({
                    "words": sorted_words,
                    "page": page
                })
        return candidates
    
    def find_best_match(self, target_text, candidates):
        best_match = None
        best_score = 0
        target_clean = target_text.replace(" ", "").lower()  

        for candidate in candidates:
            words = candidate['words']
            page = candidate['page']
            n = len(words)

            for i in range(n):
                for j in range(i + 1, n + 1):
                    phrase_words = words[i:j]
                    raw_phrase = ''.join([w['text'] for w in phrase_words])
                    display_phrase = ' '.join([w['text'] for w in phrase_words])
                    raw_phrase_clean = raw_phrase.replace(" ", "").lower()

                    score = Levenshtein.ratio(raw_phrase_clean, target_clean) * 100

                    if score > best_score:
                        merged_bbox = self.merge_bounding_boxes([w['boundingPolygon'] for w in phrase_words])
                        best_score = score
                        best_match = {
                            "matched_text": display_phrase,
                            "score": score,
                            "boundingPolygon": merged_bbox,
                            "page": page
                        }

        return best_match

    def safe_split(self, value):
        """Split by comma and clean up."""
        if isinstance(value, str) and value.strip().lower() != "none":
            return [v.strip() for v in value.split(',') if v.strip()]
        return []

    def extract_box(self, ocr_data, extracted_df, country_name_eng):
        read_results = ocr_data["readResult"]["blocks"]

        expiry_value = extracted_df["certificate_expiry_date"].iloc[0]
        validity_value = extracted_df["certificate_validity_in_years"].iloc[0]
        if expiry_value == "None":
            expiry_date = validity_value if validity_value != "None" else extracted_df["validity_in_months"].iloc[0]
        else:
            expiry_date = expiry_value

        model_number = extracted_df["Model_number"].iloc[0]
        country_name = extracted_df["country_name"].iloc[0]
        issue_date = extracted_df["certificate_issue_date"].iloc[0]

        words = self.extract_words(read_results)
        candidates = self.group_words(words, y_threshold=10)

        labeled_polygons = []

        if isinstance(expiry_date, str) and expiry_date.strip().lower() != "none":
            match_expiry = self.find_best_match(expiry_date, candidates)
            if match_expiry:
                labeled_polygons.append({
                    "label": "Certification expiry date",
                    "polygon": match_expiry['boundingPolygon'],
                    "page": match_expiry['page']
                })
        else:
            labeled_polygons.append({
                "label": "Certification expiry date",
                "polygon": [{"x": 0, "y": 0}] * 4,
                "page": 0
            })
            
        if isinstance(issue_date, str) and issue_date.strip().lower() != "none":
            match_expiry = self.find_best_match(issue_date, candidates)
            if match_expiry:
                labeled_polygons.append({
                    "label": "Last approval date",
                    "polygon": match_expiry['boundingPolygon'],
                    "page": match_expiry['page']
                })
        else:
            labeled_polygons.append({
                "label": "Last approval date",
                "polygon": [{"x": 0, "y": 0}] * 4,
                "page": 0
            })

        model_values = self.safe_split(model_number)
        if model_values:
            for val in model_values:
                match = self.find_best_match(val, candidates)
                if match:
                    labeled_polygons.append({
                        "label": "Model code",
                        "polygon": match['boundingPolygon'],
                        "page": match['page']
                    })
        else:
            labeled_polygons.append({
                "label": "Model code",
                "polygon": [{"x": 0, "y": 0}] * 4,
                "page": 0
            })
            
        country_values = self.safe_split(country_name)
        if country_values:
            for val in country_values:
                match = self.find_best_match(val, candidates)
                if match:
                    labeled_polygons.append({
                        "label": f"Country (in English - {country_name_eng})",
                        "polygon": match['boundingPolygon'],
                        "page": match['page']
                    })
        else:
            labeled_polygons.append({
                "label": "Country",
                "polygon": [{"x": 0, "y": 0}] * 4,
                "page": 0
            })

        return labeled_polygons