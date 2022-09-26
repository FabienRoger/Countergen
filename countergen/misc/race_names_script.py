"""Use data & code from https://github.com/lead-ratings/gender-guesser.

Copy and paste the data folder next to the script to make it run"""

import os.path
import codecs


class NoCountryError(Exception):
    """Raised when non-supported country is queried"""

    pass


class Detector:
    """Get gender by first name"""

    COUNTRIES = """great_britain ireland usa italy malta portugal spain france
                   belgium luxembourg the_netherlands east_frisia germany austria
                   swiss iceland denmark norway sweden finland estonia latvia
                   lithuania poland czech_republic slovakia hungary romania
                   bulgaria bosniaand croatia kosovo macedonia montenegro serbia
                   slovenia albania greece russia belarus moldova ukraine armenia
                   azerbaijan georgia the_stans turkey arabia israel china india
                   japan korea vietnam other_countries
                 """.split()

    def __init__(self, case_sensitive=True):

        """Creates a detector parsing given data file"""
        self.case_sensitive = case_sensitive
        self._parse(os.path.join(os.path.dirname(__file__), "data/nam_dict.txt"))

    def _parse(self, filename):
        """Opens data file and for each line, calls _eat_name_line"""
        self.names = {}
        with codecs.open(filename, encoding="utf-8") as f:
            for line in f:
                self._eat_name_line(line.strip())

    def _eat_name_line(self, line):
        """Parses one line of data file"""
        if line[0] not in "#=":
            parts = line.split()
            country_values = line[30:-1]
            name = parts[1]
            if not self.case_sensitive:
                name = name.lower()

            if parts[0] == "M":
                self._set(name, "male", country_values)
            elif parts[0] == "1M" or parts[0] == "?M":
                self._set(name, "mostly_male", country_values)
            elif parts[0] == "F":
                self._set(name, "female", country_values)
            elif parts[0] == "1F" or parts[0] == "?F":
                self._set(name, "mostly_female", country_values)
            elif parts[0] == "?":
                self._set(name, "andy", country_values)
            else:
                raise "Not sure what to do with a sex of %s" % parts[0]

    def _set(self, name, gender, country_values):
        """Sets gender and relevant country values for names dictionary of detector"""
        if "+" in name:
            for replacement in ["", " ", "-"]:
                self._set(name.replace("+", replacement), gender, country_values)
        else:
            if name not in self.names:
                self.names[name] = {}
            self.names[name][gender] = country_values

    def _most_popular_gender(self, name, counter):
        """Finds the most popular gender for the given name counting by given counter"""
        if name not in self.names:
            return "unknown"

        max_count, max_tie = (0, 0)
        best = list(self.names[name].keys())[0]
        for gender, country_values in list(self.names[name].items()):
            count, tie = counter(country_values)
            if count > max_count or (count == max_count and tie > max_tie):
                max_count, max_tie, best = count, tie, gender

        return best if max_count > 0 else "andy"

    def get_gender(self, name, country=None):
        """Returns best gender for the given name and country pair"""
        if not self.case_sensitive:
            name = name.lower()

        if name not in self.names:
            return "unknown"
        elif not country:

            def counter(country_values):
                country_values = list(map(ord, country_values.replace(" ", "")))
                return (len(country_values), sum([c > 64 and c - 55 or c - 48 for c in country_values]))

            return self._most_popular_gender(name, counter)
        elif country in self.__class__.COUNTRIES:
            index = self.__class__.COUNTRIES.index(country)
            counter = lambda e: (ord(e[index]) - 32, 0)
            return self._most_popular_gender(name, counter)
        else:
            raise NoCountryError("No such country: %s" % country)


d = Detector()

west_countries = """great_britain ireland usa italy portugal spain france
                   belgium luxembourg the_netherlands east_frisia germany austria
                   swiss iceland denmark norway sweden finland
                 """.split()
asian_countries = """armenia azerbaijan georgia the_stans turkey arabia israel
                   japan korea vietnam china india
                   """.split()


def parse(c):
    if c == " ":
        return 0
    return ord(c) - ord("0")


def get_name_counts(countries, gender, top_k=100):
    country_idxs = [d.COUNTRIES.index(c) for c in countries]

    def get_count(cv):
        if gender not in cv:
            return 0
        return sum(2 ** (parse(cv[gender][i])) for i in country_idxs)  # sum of popularity

    names_counts = [(n, get_count(country_values)) for n, country_values in d.names.items()]
    names_counts.sort(key=lambda t: -t[1])  # Most popular first
    return [n for n, c in names_counts[:top_k]]


names = get_name_counts(west_countries, "female", top_k=500)
print(" ".join(names))
print("\n")
names = get_name_counts(asian_countries, "female", top_k=500)
print(" ".join(names))
print("\n")
names = get_name_counts(west_countries, "male", top_k=500)
print(" ".join(names))
print("\n")
names = get_name_counts(asian_countries, "male", top_k=500)
print(" ".join(names))
print("\n")
