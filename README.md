# Narendra Kumar Vadapalli - website

![](https://img.shields.io/badge/version-1.1.1-green.svg) ![](https://img.shields.io/badge/License-MIT-orange.svg)

This is my personal website that is based off of the [Gatsby Elemental Starter](https://github.com/akzhy/gatsby-theme-elemental)

### Getting Started

```
git clone git@github.com:narenandu/narenvadapalli_dot_com.git
cd narenvadapalli_dot_com
npm install
gatsby develop   # this will spin up the webpage at 8080 by default
```

### Configuring

Almost all features of this starter are editable. In order to personalize, open the `gatsby-config.js` file and start editing the below part.

```javascript
// Do not remove any of the properties below.

let siteMetadata = {
    title: `Narendra Kumar Vadapalli`, // Title of your webpage
    capitalizeTitleOnHome: true, // Whether to capitalize the letter on homepage
    logo: `/images/logo.png`, // Logo
    icon: `/images/icon.png`, // Favicon, shown in the browsers "tab"
    titleImage: `/images/wall.jpg`, // The main title is filled with an image.
    introTag: `PROGRAMMER | VISUAL DESIGN ENTHUSIAST`, // Intro tag shown below title
    description: `Personl Website`,
    author: `@narenandu`,
    blogItemsPerPage: 10,
    portfolioItemsPerPage: 10,
    darkmode: true, // Whether to enable the darkmode. Change to false if you want the light mode
    switchTheme: true, // Whether to show a switch theme button on the navbar
    // The links shown on the navbar and footer, follow the same structure to add or remove more items.
    navLinks: [
        {
            name: "HOME",
            url: "/"
        },
        {
            name: "ABOUT",
            url: "/about"
        },
        {
            name: "BLOG",
            url: "/blog"
        },
        {
            name: "PORTFOLIO",
            url: "/portfolio"
        },
        {
            name: "CONTACT",
            url: "/contact"
        }
    ],
    // Same as navbar links, except these are shown on the footer
    footerLinks: [],
    // Your social profile links. The icons of the given social medias are available in the static folder. If you are adding a new item, include the icon in the static/images folder.
    social: [
        {
            name: "imdb",
            icon: "/images/Facebook.svg",
            url: "#"
        },
        {
            name: "Twitter",
            icon: "/images/Twitter.svg",
            url: "#"
        },
        {
            name: "Instagram",
            icon: "/images/Instagram.svg",
            url: "#"
        },
        {
            name: "Youtube",
            icon: "/images/Youtube.svg",
            url: "#"
        }
    ],
    contact: {
        /* Leave this completely empty (no space either) if you don't want a contact form. */
        api_url: "./test.json",
        description: `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed sit amet accumsan arcu. Proin ac consequat arcu.`,
        mail: "",
        phone: "",
        address: ""
    }
};
```

#### Creating new blog posts.

Open the `contents/blog` folder and create a new folder with the name you wish to see as the URL. Inside the folder create an `index.md` file and also include any files you wish to add.

The frontmatter should be of the below structure

```
---
title: Title of your post
date: 2019-06-29 <-- Date should be in the given format
template: blog <-- The template you wish to use. "blog" for blog posts
image: ./image.jpg <-- Image shown on the list pages and also used as open graph image
banner: ./banner.jpg <-- Banner shown in the blog post
description: The description shown in the listing page. Also used for SEO description.
---
```

If you don't want the blog section, simply delete everything inside the `contents/blog` folder. (Do not delete the folder itself)

#### Creating new portfolio posts.

Open the `contents/portfolio` folder and create a new folder with the name you wish to see as the URL. Inside the folder create an `index.md` file and also include any files you wish to add.

The frontmatter should be of the below structure

```
---
title: Title of your post
date: 2019-06-29 <-- Date should be in the given format
template: blog <-- The template you wish to use. "blog" for blog posts
image: ./image.jpg <-- Image shown on the list pages and also used as open graph image
description: The description shown in the listing page. Also used for SEO description.
---
```

Portfolio pages support the creation of grids.

To create a grid, follow the below structure

```
[row]
[col]
**Markdown**
[/col]
[/row]
```

The columns will have equal width on wide screens, and will expand on smaller screens.

#### Creating miscellaneous posts

These posts follow the URL structure of `http://example.com/miscellaneous-post/`. They are useful for creating pages like `privacy-policy`

The "About" page is created as a miscellaneous post.

## Contributing

Any kind of contributions are welcome. Bump the version and create a PR.
