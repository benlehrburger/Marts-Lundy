# Make a donor object in the form of { #ID: { categories }}
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

    def makeDonor(self):
        # Format into a dictionary of dictionaries with ID as key
        donor = {}

        donor[str(self.entity.cell_value(self.rows, 0))] = {category: [] for category in self.categories}

        # Add info from entity table for each donor
        for category in self.entity_cats:
            donor[str(self.entity.cell_value(self.rows, 0))][category] = self.entity.cell_value(self.rows, self.entity_cats.index(category)+1)

        # Add info for other tables if donor's ID exists
        def itemize(table, ids, cats, skipper):
            if int(self.entity.cell_value(self.rows, 0)) in ids:
                for category in cats:
                    for r in ids[int(self.entity.cell_value(self.rows, 0))]:
                        donor[str(self.entity.cell_value(self.rows, 0))][category].append(table.cell_value(r, cats.index(category)+skipper))

        # *** FOR EACH TABLE ***
        # skipper: how many categories are we skipping over? ID? Name?
        itemize(self.honors, self.honors_ids, self.honors_cats, 1)
        itemize(self.gifts, self.gift_ids, self.gifts_cats, 1)

        return donor
