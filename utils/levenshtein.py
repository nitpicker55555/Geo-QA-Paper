def levenshtein_distance(s1, s2):

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def are_strings_similar(s1, s2, threshold_percentage=25):
    max_length = max(len(s1), len(s2))
    threshold = max_length * (threshold_percentage / 100)
    distance = levenshtein_distance(s1, s2)
    return distance <= threshold

# # Example usage
# s1 = "strawberries"
# s2 = "strawberry"
# result = are_strings_similar(s1, s2, 25)  # Checking if the distance is within 20% of the longest string
# print(result)
