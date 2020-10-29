import React from "react";
import SectionTitle from "./sectiontitle";
import { StaticQuery, graphql } from "gatsby";
import SocialLinks from "./sociallinks";
import "../style/contact.less";
import { Document, Page } from 'react-pdf';

class Resume extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            submitDisabled: false
        };

        this.showContactForm = false;

        if (this.props.contact.api_url === "") {
            this.showContactForm = false;
        }
    }

    render() {
        return (
            <section id="contact" className="container">
                <div className="section-title">
                    <SectionTitle title="RESUME" />
                </div>
                <div>
                    <Document
                        file="../../static/pdf/Narendra_Kumar_Vadapalli_Resume.pdf"
                    >
                        <Page pageNumber={1} />
                    </Document>

                </div>
                <div
                    className={"row" + (this.showContactForm ? "" : " no-form")}
                    ref={c => (this.contactArea = c)}
                >
                    {this.showContactForm && <div className="col s12 m6"></div>}
                    <div
                        className={
                            this.showContactForm
                                ? "col s12 m6 details"
                                : "col s12 details"
                        }
                    >

                        <ul>
                            <li>
                                <SocialLinks />
                            </li>
                        </ul>
                    </div>
                </div>
            </section>
        );
    }
}

export default () => (
    <StaticQuery
        query={graphql`
            query {
                site {
                    siteMetadata {
                        contact {
                            api_url
                            description
                            mail
                            phone
                            address
                        }
                    }
                }
            }
        `}
        render={data => <Resume contact={data.site.siteMetadata.contact} />}
    />
);
