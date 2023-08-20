# Exercise #9. Python Programming
import numpy as np
import matplotlib.pyplot as plt

#########################################
# Question A1 - do not delete this comment
#########################################


def analyze_rating_data(filename):
    rating = np.loadtxt(filename, delimiter=',')

    # The number of seasons (number of rows in the table)
    print('The number of seasons:')
    print(rating.shape[0])
    # The highest rating recorded in the entire file for an episode (maximum value in the table)
    print('The highest rating ever recorded for an episode:')
    print(rating.max())

    # Average rating for the first episode over all seasons (average of the first column)
    print('Average rating for the first episode over all seasons:')
    print(rating[:, 0].mean(axis=0))

    # Number of episodes which had a rating lower than 8 (how many values lower than 8 exist in the table)
    print('Number of episodes which had a rating lower than 8:')
    print(rating[rating < 8].size)

    # Is there at least one episode with a rating of 15 ? Print True or False without using IF
    print('Is there at least one episode with a rating of 15:')
    print(rating[rating == 15].any())

    # What is the maximal total season rating ? (sum the rating in each season and print the maximum over the sums)
    print('The maximal total season rating:')
    print(rating.sum(axis=1).max())

    # Print a vector holding the minimal rating for each episode (vector of column minimums)
    print('Minimal rating for each episode:')
    print(rating.min(axis=0))

# Testing your code -------------------------------
# print ('Output for ISRAEL:----------------------')
# analyze_rating_data('Israel.csv')
# print ('Output for SPAIN:-----------------------')
# analyze_rating_data('Spain.csv')


#########################################
# Question B1 - do not delete this comment
#########################################
def load_covid_world_matrix(filename, fieldname):
    try:
        f = np.loadtxt(filename, dtype=str, delimiter=',')
        header = f[0]
        i = np.argwhere(header == fieldname)
        i = i.sum()
        field = f[1:, i]
        continent = f[1:, 1]
        countries = f[1:, 2]
        dates = f[1:, 3]
        countries_uni = np.unique(countries[continent != ""])
        dates_uni = np.unique(dates)
        field[field == ""] = '0'
        matrix = np.zeros((len(countries_uni), len(dates_uni)), dtype=float)
        i = 0
        for country in countries_uni:
            mask = countries == country
            n = dates[mask]
            m = field[mask]
            m = m.astype(float)
            n = np.isin(dates_uni, n)
            matrix[i][n] = m
            i += 1
        return countries_uni, dates_uni, matrix
    except IOError:
        print('IOError encountered')


#########################################
# Question B2 - do not delete this comment
#########################################
def analyze_covid_data(countries, dates, matrix):
    print('Are there any negative values in the table?')
    print(np.any(matrix < 0))

    print('In how many days more than 8000 new cases were identified in Israel?')
    print(np.count_nonzero(np.argwhere(matrix[countries == 'Israel'] > 8000)))

    print('Number of countries with more than 1 million total cases:')
    print(np.count_nonzero(matrix.sum(axis=1) > 1000000))

    print('Name of country with the highest total number of daily cases in the first 30 days appearing in the table:')
    print(countries[np.sum(np.argmax(matrix[:, 0:30].sum(axis=1)))])

    print('Date with maximal number of new cases in all countries together:')
    print(dates[np.sum(np.argmax(matrix.sum(axis=0)))])


#########################################
# Question B3.1 - do not delete this comment
#########################################
def plot_country_data(matrix, countries, country):
    c = np.argwhere(countries == country)
    c = c.sum()
    x = matrix[c, :]
    plt.xlabel('Days')
    plt.title(country)
    plt.plot(x, 'r')
    plt.show()


#########################################
# Question B3.2 - do not delete this comment
#########################################
def plot_top_countries(matrix, countries):
    top = np.flip((-matrix.sum(axis=1)).argsort()[:5])
    count = countries[top]
    top_data = np.transpose(matrix[top, :])
    plt.xlabel('Days')
    plt.title('Top 5 countries')
    lines = plt.plot(top_data)
    plt.legend(lines, count, loc=1, fontsize=8)
    plt.show()


#########################################
# Question B3.3 - do not delete this comment
#########################################
def draw_covid_heatmap(matrix, countries):
    small = matrix[:, 0:100]
    small[small < 1] = 1
    small = np.log2(small)
    temp = (-small.sum(axis=1)).argsort()[:20]
    top = matrix[temp, :]
    top[top < 1] = 1
    top = np.log2(top)
    count = countries[temp]
    im = plt.imshow(top, cmap='afmhot', aspect='auto', interpolation='none', origin='lower')
    plt.yticks(ticks=range(0, count.size), labels=count)
    plt.colorbar(im)
    plt.show()


# Testing your code -------------------------------
countries, dates, matrix = load_covid_world_matrix('owid-covid-data.csv', 'new_cases')
# np.savetxt("matrix_new_cases.csv", matrix, delimiter=",", fmt='%f')
# np.savetxt("dates_new_cases.csv", dates, delimiter=",", fmt='%s')
# np.savetxt("countries_new_cases.csv", countries, delimiter=",", fmt='%s')

print(countries.shape)
print(dates.shape)
print(matrix.shape)
print(matrix.sum())

analyze_covid_data(countries, dates, matrix)

plot_country_data(matrix, countries, 'Israel')
plot_top_countries(matrix, countries)
draw_covid_heatmap(matrix, countries)
