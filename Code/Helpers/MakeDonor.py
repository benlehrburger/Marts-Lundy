# Ben Lehrburger
# Philanthropy in higher education classifier

# Build a donor object
# Key: ID, Values: dictionary with key=category and value=value

# Wrap a donor object
class Donor:

    def __init__(self, rows, entity, honors, gifts, honors_ids, gift_ids, entity_cats, honors_cats, gifts_cats, categories):

        self.rows = rows
        self.entity = entity
        self.honors = honors
        self.gifts = gifts

        self.honors_ids = honors_ids
        self.gift_ids = gift_ids

        self.entity_cats = entity_cats
        self.honors_cats = honors_cats
        self.gifts_cats = gifts_cats
        self.categories = categories

        self.entity_row = range(0, rows)
        self.honors_row = range(0, rows)
        self.gifts_row = range(0, rows)

    # Fill each categorical value
    def makeDonor(self):

        # Format into a dictionary of dictionaries with ID as key
        donor = {}

        # Add categories
        donor[str(self.entity.cell_value(self.rows, 0))] = {category: [] for category in self.categories}

        # Add info from entity table for each donor
        for category in self.entity_cats:
            donor[str(self.entity.cell_value(self.rows, 0))][category] = self.entity.cell_value(self.rows, self.entity_cats.index(category)+1)

        # Add data from honors table
        self.itemize(donor, self.honors, self.honors_ids, self.honors_cats, 1)

        # Add data from gifts table
        self.itemize(donor, self.gifts, self.gift_ids, self.gifts_cats, 1)

        return donor

    # Add info for other tables if donor's ID exists
    # Skipper: which categories do we not want to include?
    def itemize(self, donor, table, ids, cats, skipper):

        if int(self.entity.cell_value(self.rows, 0)) in ids:
            for category in cats:
                for r in ids[int(self.entity.cell_value(self.rows, 0))]:

                    donor[str(self.entity.cell_value(self.rows, 0))][category].append(
                        table.cell_value(r, cats.index(category) + skipper))
