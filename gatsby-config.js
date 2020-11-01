let siteMetadata = {
    title: `Narendra Kumar Vadapalli`,
    capitalizeTitleOnHome: true,
    logo: `/images/logo.png`,
    icon: `/images/icon.png`,
    titleImage: `/images/wall.jpg`,
    introTag: `PROGRAMMER | VISUAL Design Enthusiast`,
    description: `Person with a balanced left and right brain`,
    author: `@`,
    blogItemsPerPage: 10,
    portfolioItemsPerPage: 10,
    darkmode: true,
    switchTheme: true,
    navLinks: [
        {
            name: "HOME",
            url: "/"
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
            name: "RESUME",
            url: "/resume"
        },
        {
            name: "ABOUT",
            url: "/about"
        },
        {
            name: "CONTACT",
            url: "/contact"
        }
    ],
    footerLinks: [
        {
            name: "GitHub",
            url: "https://github.com/narenandu/narenvadapalli_dot_com"
        }
    ],
    social: [
        {
            name: "Twitter",
            icon: "/images/twitter.svg",
            url: "https://twitter.com/narenandu"
        },
        {
            name: "LinkedIn",
            icon: "/images/linkedin.svg",
            url: "https://www.linkedin.com/in/narenandu/"
        },
        {
            name: "IMDB",
            icon: "/images/imdb.svg",
            url: "https://www.imdb.com/name/nm4511578/"
        },
        {
            name: "Github",
            icon: "/images/github.svg",
            url: "https://github.com/narenandu"
        },
        {
            name: "CodeMentor",
            icon: "https://cdn.codementor.io/badges/i_am_a_codementor_dark.svg",
            url:
                "https://www.codementor.io/narenandu?utm_source=github&utm_medium=button&utm_term=narenandu&utm_campaign=github"
        }
    ],
    contact: {
        /* Leave the below value completely empty (no space either) if you don't want a contact form. */
        api_url: "./test.json",
        description: `Reach out to me on social media`,
        mail: "",
        phone: "",
        address: ""
    }
};

module.exports = {
    siteMetadata: siteMetadata,
    plugins: [
        `gatsby-plugin-sharp`,
        `gatsby-transformer-sharp`,
        `gatsby-plugin-react-helmet`,
        {
            resolve: `gatsby-transformer-remark`,
            options: {
                plugins: [
                    "gatsby-remark-copy-linked-files",
                    {
                        resolve: `gatsby-remark-embed-video`,
                        options: {
                            maxWidth: 800,
                            ratio: 1.77,
                            height: 400,
                            related: false,
                            noIframerder: true
                        }
                    },
                    {
                        resolve: `gatsby-remark-images`,
                        options: {
                            maxWidth: 590
                        }
                    },
                    {
                        resolve: `gatsby-remark-responsive-iframe`,
                        options: {
                            wrapperStyle: `margin-bottom: 1.0725rem`
                        }
                    }
                ]
            }
        },
        {
            resolve: `gatsby-source-filesystem`,
            options: {
                name: `contents`,
                path: `${__dirname}/contents/`
            }
        },
        {
            resolve: `gatsby-plugin-less`,
            options: {
                strictMath: true
            }
        },
        {
            resolve: `gatsby-plugin-google-analytics`,
            options: {
                trackingId:
                    process.env.GATSBY_GOOGLE_ANALYTICS_TRACKING_ID || "none"
            }
        },
        {
            resolve: 'gatsby-plugin-web-font-loader',
            options: {
                    google: {
                    families: ['Ubuntu', 'Roboto', 'Maven Pro', 'Droid Sans', 'Droid Serif']
                }
            }
        }
    ]
};
