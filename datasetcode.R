install.packages(c("tidyverse", "mice", "DMwR2", "recommenderlab", "ggplot2", "VIM"))
library(tidyverse)
library(mice)       # For MICE imputation
library(DMwR2)      # For k-NN imputation
library(ggplot2)    # For visualization
library(VIM)        # For missing data patterns
ratings <- read.table("ml-100k/u.data", 
                      header = FALSE, 
                      sep = "\t", 
                      col.names = c("userId", "movieId", "rating", "timestamp")
                      movies <- read.table("ml-100k/u.item", 
                                           header = FALSE, 
                                           sep = "|", 
                                           quote = "",
                                           col.names = c("movieId", "title", "releaseDate", "videoReleaseDate",
                                                         "IMDbURL", "unknown", "Action", "Adventure", "Animation",
                                                         "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                                         "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi",
                                                         "Thriller", "War", "Western"))
                      cat("Missing values in ratings:", sum(is.na(ratings)), "\n")
                      cat("Missing values in movies:", sum(is.na(movies)), "\n")
                      # Convert to a user-item matrix (for missing pattern visualization)
                      ratings_matrix <- ratings %>%
                        select(userId, movieId, rating) %>%
                        spread(key = movieId, value = rating)
                      
                      # Visualize missingness pattern
                      VIM::aggr(ratings_matrix, numbers = TRUE, sortVars = TRUE)
                      set.seed(123)
                      missing_mask <- sample(nrow(ratings), size = 0.1 * nrow(ratings), replace = FALSE)
                      ratings_with_NA <- ratings
                      ratings_with_NA$rating[missing_mask] <- NA
                      
                      # Verify
                      cat("Artificially introduced missing values:", sum(is.na(ratings_with_NA$rating)), "\n")
                      mean_imputed <- ratings_with_NA %>%
                        mutate(rating_imputed = ifelse(is.na(rating), mean(rating, na.rm = TRUE), rating))
                      
                      median_imputed <- ratings_with_NA %>%
                        mutate(rating_imputed = ifelse(is.na(rating), median(rating, na.rm = TRUE), rating))
                      library(dplyr)
                      library(Matrix)        # For creating and working with sparse matrices
                      library(recommenderlab) # For recommendation system specific structures & algorithms
                      
                      # --- 2. Simulate/Load Data ---
                      # Option A: Simulate Sparse Data (if you don't have a ratings file)
                      set.seed(123) # for reproducibility
                      n_users <- 1000
                      n_movies <- 2000
                      n_ratings <- 50000 # Much less than n_users * n_movies
                      
                      ratings_df <- data.frame(
                        userId = sample(1:n_users, n_ratings, replace = TRUE),
                        movieId = sample(1:n_movies, n_ratings, replace = TRUE),
                        rating = sample(1:5, n_ratings, replace = TRUE, prob = c(0.1, 0.1, 0.2, 0.3, 0.3)) # Skew towards higher ratings
                      )
                      # Remove potential duplicate user-movie ratings (keep first)
                      ratings_df <- ratings_df %>% distinct(userId, movieId, .keep_all = TRUE)
                      n_ratings <- nrow(ratings_df) # Update actual number of ratings
                      
                      print("Simulated Data Head:")
                      head(ratings_df)
                      cat("\n")
                      
                      # Option B: Load Data (if you have a ratings.csv file)
                      # Make sure the file is in your working directory or provide the full path
                      # tryCatch({
                      #   ratings_df <- read.csv("ratings.csv") # Adjust filename if needed
                      #   # Ensure column names match (e.g., userId, movieId, rating)
                      #   # ratings_df <- ratings_df %>% select(userId, movieId, rating) # Select necessary columns
                      #   n_users <- length(unique(ratings_df$userId))
                      #   n_movies <- length(unique(ratings_df$movieId))
                      #   n_ratings <- nrow(ratings_df)
                      #   print("Loaded Data Head:")
                      #   head(ratings_df)
                      # }, error = function(e) {
                      #   stop("Error loading ratings data: ", e$message)
                      # })
                      # cat("\n")
                      
                      
                      # --- 3. Calculate Sparsity ---
                      total_possible_ratings <- n_users * n_movies
                      sparsity <- 1 - (n_ratings / total_possible_ratings)
                      
                      print(paste("Number of Users:", n_users))
                      print(paste("Number of Movies:", n_movies))
                      print(paste("Number of Ratings:", n_ratings))
                      print(paste("Total Possible Ratings:", total_possible_ratings))
                      print(paste("Sparsity:", round(sparsity * 100, 2), "%"))
                      cat("\n")
                      
                      
                      # --- 4. Efficient Representation: Sparse Matrix ---
                      # Most recommendation algorithms work better with sparse matrices directly.
                      # The 'Matrix' package provides efficient structures.
                      
                      # Map userIds and movieIds to consecutive integers if they aren't already
                      # (Our simulated data uses consecutive IDs, but real data might not)
                      user_map <- data.frame(userId = sort(unique(ratings_df$userId)), user_idx = 1:n_users)
                      movie_map <- data.frame(movieId = sort(unique(ratings_df$movieId)), movie_idx = 1:n_movies)
                      
                      # Join maps to get indices
                      ratings_mapped <- ratings_df %>%
                        inner_join(user_map, by = "userId") %>%
                        inner_join(movie_map, by = "movieId")
                      
                      # Create the sparse matrix: users as rows, movies as columns
                      # Using triplet format (row index, column index, value)
                      sparse_ratings <- sparseMatrix(
                        i = ratings_mapped$user_idx,
                        j = ratings_mapped$movie_idx,
                        x = ratings_mapped$rating,
                        dims = c(n_users, n_movies),
                        dimnames = list(Users = user_map$userId, Movies = movie_map$movieId)
                      )
                      
                      print("Sparse Matrix Details:")
                      print(dim(sparse_ratings))
                      print(object.size(sparse_ratings), units = "Mb") # Show memory usage
                      print("First 5x10 slice of the sparse matrix:")
                      print(sparse_ratings[1:5, 1:10])
                      cat("\n")
                      
                      
                      # --- 5. Filtering (Reducing Sparsity) ---
                      # WARNING: This removes data. Apply judiciously based on your goals.
                      # Common practice: Remove users/movies with too few ratings.
                      
                      min_ratings_per_user <- 5
                      min_ratings_per_movie <- 5
                      
                      print(paste("Filtering: Keeping users with >=", min_ratings_per_user,
                                  "ratings and movies with >=", min_ratings_per_movie, "ratings."))
                      
                      # Filter based on user counts
                      user_counts <- ratings_df %>% count(userId)
                      users_to_keep <- user_counts %>% filter(n >= min_ratings_per_user) %>% pull(userId)
                      ratings_filtered_users <- ratings_df %>% filter(userId %in% users_to_keep)
                      
                      # Filter based on movie counts (using the already user-filtered data)
                      movie_counts <- ratings_filtered_users %>% count(movieId)
                      movies_to_keep <- movie_counts %>% filter(n >= min_ratings_per_movie) %>% pull(movieId)
                      ratings_filtered <- ratings_filtered_users %>% filter(movieId %in% movies_to_keep)
                      
                      # Update counts and recalculate sparsity
                      n_users_filtered <- length(unique(ratings_filtered$userId))
                      n_movies_filtered <- length(unique(ratings_filtered$movieId))
                      n_ratings_filtered <- nrow(ratings_filtered)
                      total_possible_filtered <- n_users_filtered * n_movies_filtered # Use filtered dimensions
                      sparsity_filtered <- 1 - (n_ratings_filtered / total_possible_filtered) # Recalculate based on filtered space
                      
                      print(paste("Users remaining after filtering:", n_users_filtered))
                      print(paste("Movies remaining after filtering:", n_movies_filtered))
                      print(paste("Ratings remaining after filtering:", n_ratings_filtered))
                      print(paste("New Sparsity (within filtered space):", round(sparsity_filtered * 100, 2), "%"))
                      cat("\n")
                      
                      
                      print("--- Sparsity Handling Demonstration Complete ---")
                      library(dplyr)
                      library(stringr)
                      library(ggplot2) # For visualization (boxplots)
                      
                      # --- 1. Load the Data ---
                      # Make sure 'movie.csv' is in your R working directory or provide the full path.
                      tryCatch({
                        movies <- read.csv("movie.csv", stringsAsFactors = FALSE, quote = "\"")
                        print("Data loaded successfully.")
                        head(movies)
                        str(movies)
                      }, error = function(e) {
                        stop("Error loading movie.csv: ", e$message,
                             "\nMake sure the file exists and is in the correct directory.")
                      })
                      
                      # --- 2. Feature Engineering ---
                      # Extract Year from Title and Count Number of Genres
                      
                      movies <- movies %>%
                        mutate(
                          # Extract year using regex: look for 4 digits within parentheses
                          year_match = str_extract(title, "\\((\\d{4})\\)"),
                          # Remove parentheses and convert to numeric year
                          year = as.numeric(str_sub(year_match, 2, -2)),
                          # Count genres by splitting the string and counting elements
                          # Handle '(no genres listed)' explicitly as 0 genres
                          num_genres = ifelse(genres == "(no genres listed)" | genres == "", 0,
                                              sapply(strsplit(genres, "\\|"), length))
                        )
                      
                      # Display first few rows with new columns
                      print("--- Data with Engineered Features ---")
                      head(movies)
                      cat("\n")
                      
                      # --- 3. Numerical Outlier Detection ---
                      
                      # A. Outliers based on Year (using IQR method)
                      #-------------------------------------------
                      print("--- Outliers based on Year ---")
                      # Calculate IQR bounds, ignoring potential NAs in year extraction
                      Q1_year <- quantile(movies$year, 0.25, na.rm = TRUE)
                      Q3_year <- quantile(movies$year, 0.75, na.rm = TRUE)
                      IQR_year <- Q3_year - Q1_year
                      lower_bound_year <- Q1_year - 1.5 * IQR_year
                      upper_bound_year <- Q3_year + 1.5 * IQR_year
                      
                      outliers_year <- movies %>%
                        filter(!is.na(year) & (year < lower_bound_year | year > upper_bound_year))
                      
                      print(paste("Year IQR:", IQR_year, " | Lower Bound (Q1 - 1.5*IQR):", round(lower_bound_year, 2),
                                  " | Upper Bound (Q3 + 1.5*IQR):", round(upper_bound_year, 2)))
                      
                      if (nrow(outliers_year) > 0) {
                        print("Potential Year Outliers (outside 1.5 * IQR):")
                        print(outliers_year %>% select(movieId, title, year))
                      } else {
                        print("No significant year outliers found using the IQR method.")
                      }
                      
                      # Visualize Year distribution
                      boxplot(movies$year, main="Boxplot of Movie Release Years", ylab="Year", outline = TRUE)
                      title(sub = paste("Outliers beyond", round(lower_bound_year,0), "and", round(upper_bound_year, 0), "are shown as points"), cex.sub = 0.8)
                      cat("\n")
                      
                      
                      # B. Outliers based on Number of Genres (using IQR method)
                      #-------------------------------------------
                      print("--- Outliers based on Number of Genres ---")
                      Q1_genres <- quantile(movies$num_genres, 0.25, na.rm = TRUE)
                      Q3_genres <- quantile(movies$num_genres, 0.75, na.rm = TRUE)
                      IQR_genres <- Q3_genres - Q1_genres
                      lower_bound_genres <- Q1_genres - 1.5 * IQR_genres
                      upper_bound_genres <- Q3_genres + 1.5 * IQR_genres
                      
                      # We are usually more interested in movies with an unusually HIGH number of genres
                      outliers_num_genres_high <- movies %>%
                        filter(num_genres > upper_bound_genres)
                      
                      # Movies with 0 genres might also be considered 'outliers' in a sense
                      outliers_num_genres_low <- movies %>%
                        filter(num_genres == 0) # Includes '(no genres listed)' and empty strings
                      
                      
                      print(paste("Num Genres IQR:", IQR_genres, " | Lower Bound (Q1 - 1.5*IQR):", round(lower_bound_genres, 2),
                                  " | Upper Bound (Q3 + 1.5*IQR):", round(upper_bound_genres, 2)))
                      
                      if (nrow(outliers_num_genres_high) > 0) {
                        print(paste("Potential Num Genres Outliers (High - >", round(upper_bound_genres, 2), "genres):"))
                        print(outliers_num_genres_high %>% select(movieId, title, num_genres, genres))
                      } else {
                        print("No movies with an unusually high number of genres found using the IQR method.")
                      }
                      cat("\n")
                      
                      # Visualize Number of Genres distribution
                      boxplot(movies$num_genres, main="Boxplot of Number of Genres per Movie", ylab="Number of Genres", outline = TRUE)
                      title(sub = paste("Outliers beyond", round(upper_bound_genres, 0), "are shown as points"), cex.sub = 0.8)
                      cat("\n")
                      
                      # --- 4. Categorical/Structural Outlier Detection ---
                      
                      # C. Movies with missing or problematic years
                      #-------------------------------------------
                      missing_year_movies <- movies %>%
                        filter(is.na(year))
                      
                      print("--- Movies with Missing or Invalid Years ---")
                      if (nrow(missing_year_movies) > 0) {
                        print(paste(nrow(missing_year_movies), "movies found with missing/invalid year in title:"))
                        print(missing_year_movies %>% select(movieId, title))
                      } else {
                        print("No movies found where year extraction failed.")
                      }
                      cat("\n")
                      
                      
                      # D. Movies with empty or '(no genres listed)'
                      #-------------------------------------------
                      no_genre_movies <- movies %>%
                        filter(num_genres == 0) # Already calculated this group
                      
                      print("--- Movies with No Genres Listed or Empty Genre String ---")
                      if (nrow(no_genre_movies) > 0) {
                        print(paste(nrow(no_genre_movies), "movies found with no genres listed:"))
                        print(no_genre_movies %>% select(movieId, title, genres))
                      } else {
                        print("No movies found explicitly without genres.")
                      }
                      cat("\n")
                      
                      
                      # E. Potentially problematic Titles (e.g., very short/long - less common for outliers)
                      #-------------------------------------------
                      # Let's calculate title length (excluding the year part for consistency)
                      movies <- movies %>%
                        mutate(
                          title_no_year = str_trim(str_replace(title, "\\s*\\(\\d{4}\\)$", "")),
                          title_len = nchar(title_no_year)
                        )
                      
                      # Check summary statistics for title length
                      print("--- Title Length Summary Statistics ---")
                      summary(movies$title_len)
                      
                      # Example: Find titles shorter than 3 characters or longer than, say, 100
                      outliers_title_len <- movies %>%
                        filter(title_len < 3 | title_len > 100)
                      
                      print("--- Movies with Very Short (< 3) or Very Long (> 100) Titles ---")
                      if (nrow(outliers_title_len) > 0) {
                        print(paste(nrow(outliers_title_len), "movies found with unusual title lengths:"))
                        print(outliers_title_len %>% select(movieId, title, title_len))
                      } else {
                        print("No movies found with extremely short or long titles based on thresholds.")
                      }
                      cat("\n")
                      
                      # F. Identify very rare genres
                      #-------------------------------------------
                      # Split all genres into a single vector, excluding rows with no genres
                      all_genres_vector <- movies %>%
                        filter(num_genres > 0) %>%
                        pull(genres) %>%
                        strsplit("\\|") %>%
                        unlist()
                      
                      # Calculate frequencies
                      genre_counts <- sort(table(all_genres_vector), decreasing = FALSE)
                      
                      # Define "rare" - e.g., genres appearing <= 5 times in the dataset
                      rare_threshold <- 5
                      rare_genres <- names(genre_counts[genre_counts <= rare_threshold])
                      
                      print(paste("--- Rare Genres (appearing", rare_threshold, "times or fewer) ---"))
                      print(genre_counts[genre_counts <= rare_threshold])
                      cat("\n")
                      
                      # Find movies that contain *only* rare genres (might be interesting)
                      # This requires a bit more logic
                      movies_with_rare_genres <- movies %>%
                        filter(num_genres > 0) %>% # Exclude movies with no genres
                        rowwise() %>% # Process row by row
                        mutate(
                          movie_genres = list(strsplit(genres, "\\|")[[1]]), # Get genres for this movie
                          all_genres_rare = all(movie_genres %in% rare_genres) # Check if ALL are rare
                        ) %>%
                        filter(all_genres_rare) %>%
                        ungroup() # Finish rowwise operation
                      
                      print(paste("--- Movies Containing ONLY Rare Genres (where rare means appearing", rare_threshold, "times or fewer) ---"))
                      if (nrow(movies_with_rare_genres) > 0) {
                        print(paste(nrow(movies_with_rare_genres), "movies found containing only rare genres:"))
                        print(movies_with_rare_genres %>% select(movieId, title, genres))
                      } else {
                        print("No movies found containing only rare genres.")
                      }
                      cat("\n")
                      
                      
                      # --- 5. movieId checks (less about outliers, more about data integrity) ---
                      print("--- movieId Checks ---")
                      # Check if any movieIds are non-numeric (read.csv usually handles this, but good check)
                      if(any(is.na(as.numeric(movies$movieId)))) {
                        print("WARNING: Some movieIds are non-numeric or missing!")
                        print(movies[is.na(as.numeric(movies$movieId)), ])
                      } else {
                        print("All movieIds appear numeric.")
                      }
                      
                      # Check for duplicate movieIds
                      duplicates_movieId <- movies %>%
                        group_by(movieId) %>%
                                 filter(n() > 1)
                      
                      if (nrow(duplicates_movieId) > 0) {
                        print("WARNING: Duplicate movieIds found:")
                        print(duplicates_movieId)
                      } else {
                        print("No duplicate movieIds found.")
                      }
                      
                      # Check for large gaps in movieId sequence (descriptive, not strictly outliers)
                      movie_ids_numeric <- as.numeric(movies$movieId)
                      id_diffs <- diff(sort(movie_ids_numeric))
                      large_gaps <- which(id_diffs > 100) # Example threshold: gap > 100
                      
                      if (length(large_gaps) > 0) {
                        print(paste("Found", length(large_gaps), "large gaps (> 100) in movieId sequence. Examples:"))
                        sorted_ids <- sort(movie_ids_numeric)
                        for(i in head(large_gaps)) {
                          print(paste("Gap between movieId", sorted_ids[i], "and", sorted_ids[i+1]))
                        }
                      } else {
                        print("No unusually large gaps found in movieId sequence (using threshold > 100).")
                      }
                      cat("\n")
                      
                      print("--- Outlier Analysis Complete ---")
                      